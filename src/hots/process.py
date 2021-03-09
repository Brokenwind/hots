# -*- coding: utf-8 -*-
# @Time    : 26/8/2020 18:43 PM
# @Author  : WangKun
# @Email   : wangkun6536@163.com

import base64
import csv
import os
import pickle
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import unicodedata
from joblib import Parallel, delayed
from nlpyutil import Logger
from nlpyutil import preprocess as preprocore
from nlpyutil import usual_pattern
from tqdm import tqdm

from hots import sensitive
from . import common

_COLUMN_PROCESSED = 'processed'
_mds = set()
DEFAULT_CSV_COLS = 5
_logger = Logger()
# 防止opencc报递归深度问题
sys.setrecursionlimit(100000)

csv.field_size_limit(500 * 1024 * 1024)
# 过滤指定内容的文本
_FILTER_NOISE = [set(line.split()) for line in open(common.FILTER_NOISE_SENTENCE_PATH, encoding='UTF-8').readlines()]

_FILTER_NOISE_REGEX_LIST = [line.strip()
                            for line in open(common.FILTER_NOISE_REGEX_PATH, encoding='UTF-8').readlines() if line]
_FILTER_NOISE_REGEX = re.compile('|'.join(_FILTER_NOISE_REGEX_LIST))

_SENSITIVE_WORDS = sensitive.load_sensitive_words(merge=True)


@preprocore.process_iter
def preprocess_remove_unknow_paragraph(data: str, **kwargs):
    """
    移除如下无用文本：
    简体 | 繁体 | 无障碍阅读 | RSS订阅 国家税务总局 | 江西省人民政府  微博  微信  本站热词：税务缴税营改增税收增值税 首页 信息公开 新闻动态 政策文件 纳税服务 互动交流 首页 &gt;信息公开&gt;
    :param data:
    :param kwargs:
    :return:
    """
    special_glt_res = data.find('信息公开&gt;')
    if data.startswith('简体') and special_glt_res > 0:
        return data[special_glt_res + len('信息公开&gt;'):]

    return data


@preprocore.process_iter
def preprocess_unkown_words(data: str, **kwargs):
    """
    移除data中的&gt;和&lt;标识
    :param data:
    :param kwargs:
    :return:
    """
    data = data.replace('http：', 'http:')
    data = data.replace('https：', 'https:')
    data = re.sub('&gt;|&lt;', '', data)
    # [Cc]    Other, Control
    # [Cf]    Other, Format
    # [Cn] Other, Not Assigned
    # [Co] Other, Private Use
    cates = ['Cf', 'Cc', 'Cn', 'Co']
    res_data_list = []
    for chr in data:
        if unicodedata.category(chr) in cates:
            continue
        res_data_list.append(chr)

    return ''.join(res_data_list)


@preprocore.process_iter
def preprocess_squeeze_whitespace(data: str, **kwargs):
    """
    移除字和空格交替出现时的空格：
    例如： 倡 议 书 一、使用过的口罩的危害 正如小伙伴所说
    变为： 倡议书 一、使用过的口罩的危害 正如小伙伴所说
    :param data:
    :param kwargs:
    :return:
    """
    new_res = []
    for i, item in enumerate(re.split(usual_pattern.PATTERN_WHITE_SPACE, data)):
        if len(item) == 1 and len(new_res) != 0:
            new_res[-1] += item
        else:
            new_res.append(item)
    return ' '.join(new_res)


def preprocess_weibo_content(data: str, **kwargs):
    """
    处理微博内容
    :param line:
    :return:
    """
    topics = re.findall(usual_pattern.WEIBO_PATTERN_TOPIC, data)
    topic = common.SENTENCE_JOINT_CHR.join(topics)
    if len(topic) < 15:
        data = re.sub(usual_pattern.PATTERN_URL, common.SENTENCE_JOINT_CHR, data)
        if data.endswith("...全文"):
            data = data[:len(data) - 6]
        items = []
        if topic:
            items.append(topic)
        for item in data.split("//"):
            # 去除@的人物
            item = re.sub(r'@.{1,15}[:：\ ]', '', item)
            if not item:
                continue
            items.append(item)
        result = common.SENTENCE_JOINT_CHR.join(items)
    else:
        result = topic

    return result


def preprocess_chuantong_content(data: str, **kwargs):
    """
    处理传统数据
    :param line:
    :return:
    """
    # data = re.sub(usual_pattern.CHUANTONG_BOOK_MARK, common.SENTENCE_JOINT_CHR, data)
    data = re.sub(usual_pattern.CHUANTONG_SENTENCE_SPLITTER, common.SENTENCE_JOINT_CHR, data)

    return data


def cut_sentence(text, use_type="summarize"):
    """
    分句(文本摘要)
    :param sentence:str, like "大漠帝国"
    :param use_type:str, like "summarize" or "new-word-discovery"
    :return:list
    """
    if use_type == "summarize":
        re_sen = re.compile('[ #.,:;!?，。：；？！\n\r]')
    elif use_type == "new-word-discovery":
        re_sen = re.compile('[ #.,，"“”、<>（）《》{}【】:;!?。：；？！\n\r]')
    else:
        raise RuntimeError("use_type must be 'summarize' or 'new-word-discovery'")
    sentences = re_sen.split(text)
    sen_cuts = []
    for sen in sentences:
        if sen and str(sen).strip():
            sen_cuts.append(sen)
    return sen_cuts


def decode_base64(data):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.
    """
    data = data.strip()
    if not data:
        return ""
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    return base64.b64decode(data).decode("utf-8")


def is_filter_the_sentence(seg_text: str, text, filter_ratio=0.5):
    """
    是否过滤掉此文本
    :param seg_text:
    :param filter_ratio:
    :return:
    """
    words_set = set(seg_text.split())
    if len(words_set) == 0:
        return True
    insec_sens = words_set.intersection(_SENSITIVE_WORDS)
    if len(insec_sens) >= 1:
        return True
    for topic_words in _FILTER_NOISE:
        insect = words_set.intersection(topic_words)
        if len(insect) == len(topic_words) or len(insect) >= 3 or len(insect) / len(words_set) >= filter_ratio:
            return True
    if _FILTER_NOISE_REGEX.search(text):
        return True

    return False


def preprocess_filtered_full_data(content: pd.DataFrame,
                                  data_mark,
                                  clustered_groups=[],
                                  truncate_base=common.ARTICLE_MAX_LENGTH + common.FULL_FLOAT_CHR_NUM) -> dict:
    """
    处理经过锚定词过滤后的文本，这里的处理方式和预处理时的方法不一样
    """
    num_task = max(1, common.NUM_CPU)
    partial_len = int(np.ceil(len(content) / num_task))
    if partial_len == 0:
        partial_len = len(content)
        num_task = 1
    total_group_to_sentence = defaultdict(list)
    total_group_to_extra = defaultdict(list)
    partial_results = Parallel(n_jobs=num_task, backend="multiprocessing")(
        delayed(preprocess_filtered_full_data_single)(content.iloc[idx:idx + partial_len],
                                                      clustered_groups,
                                                      truncate_base)
        for idx in range(0, len(content), partial_len))
    # 融合每个进程处理的结果
    for partial in partial_results:
        group_to_sentence, group_to_extra = partial
        for group_key in group_to_sentence.keys():
            sentences = group_to_sentence[group_key]
            extra = group_to_extra[group_key]
            total_group_to_sentence[group_key].extend(sentences)
            total_group_to_extra[group_key].extend(extra)
    """
    name_prefix = time.strftime("%Y%m%d%H", time.localtime())
    
    with open(os.path.join(common.DATA_PROCESSED_PATH, "tmp_group_sentences_{}_{}.pkl".format(name_prefix, data_mark)),
              "wb") as f:
        pickle.dump(total_group_to_sentence, f)
    """

    return total_group_to_sentence, total_group_to_extra


def preprocess_filtered_full_data_single(content: pd.DataFrame,
                                         clustered_groups=[],
                                         truncate_base=common.ARTICLE_MAX_LENGTH + common.FULL_FLOAT_CHR_NUM,
                                         max_sentence_size=common.MAX_TOPIC_LEN) -> dict:
    """
    最后的文本进行处理
    :param truncate_base: 当文章超过300时会被截断，按照句子级别进行截断。如果此值为-1，表示不进行截断
    :return:
    """
    _weibo_content_processer = preprocore.PreprocessPipeline(func_list=[preprocess_weibo_content,
                                                                        preprocore.preprocess_replace_html_marks])
    _content_processer = preprocore.PreprocessPipeline(func_list=[preprocess_chuantong_content,
                                                                  preprocore.preprocess_replace_html_marks, ])
    content.fillna(value="", inplace=True)
    content[_COLUMN_PROCESSED] = None
    group_to_sentence_map = defaultdict(list)
    # 其它附加信息
    group_to_extra_map = defaultdict(list)
    clustered_word_set_groups = []
    for cluster_info, count in clustered_groups:
        tmp_group_set_list = []
        for group_info in cluster_info:
            group = group_info["group"]
            tmp_words = group.split("_")
            sentence_len = len(group) - len(tmp_words) + 1
            tmp_group_set_list.append((set(tmp_words), len(tmp_words), sentence_len))
        tmp_group_set_list = sorted(tmp_group_set_list, key=lambda item: (item[1], item[2]), reverse=True)
        tmp_group_set_list = [item[0] for item in tmp_group_set_list]
        clustered_word_set_groups.append(tmp_group_set_list)

    base64_error_line = 0
    # 第一行可能有文件头，也可能因为文件分割导致数据不完整
    for sentence_idx, line in tqdm(content.iterrows(), desc="processing filtered data"):
        type_text = str(line['type'])
        # 处理内容
        try:
            title_text = decode_base64(str(line["title"]))
            summary_text = decode_base64(str(line["summary"]))
            content_text = decode_base64(line["content"])
            if title_text.endswith("<--->"):
                title_text = title_text[0:len(title_text) - 5]
            title_text = "，".join(title_text.split())
        except Exception as e:
            base64_error_line += 1
            continue
        merge_content = "，".join([title_text, summary_text, content_text])
        # 先分句在处理，速度会很慢，所以这里先文章处理后在分句子
        if type_text == "wb":
            processed_content = _weibo_content_processer.process(merge_content)
        else:
            processed_content = _content_processer.process(merge_content)
        current_len = 0
        for sentence_text in cut_sentence(processed_content, use_type="summarize"):
            if len(sentence_text) < 1 or len(sentence_text) > max_sentence_size:
                continue
            seg_sentence_text = preprocore.preprocess_text_segmentation(sentence_text,
                                                                        remove_stopwords=False,
                                                                        remove_punc=False)
            if not seg_sentence_text:
                continue
            current_len += len(sentence_text)
            if truncate_base > 0 and current_len > truncate_base:
                break
            if is_filter_the_sentence(seg_sentence_text, sentence_text):
                continue
            seg_sentence_word_set = set(seg_sentence_text.split())
            is_match_cluster = False
            # 如果找到第一个匹配的word_group，就直接跳出循环
            for idx, tmp_group_set_list in enumerate(clustered_word_set_groups):
                is_match_group = False
                for group_set in tmp_group_set_list:
                    if len(group_set) > len(seg_sentence_word_set):
                        continue
                    insect = group_set.intersection(seg_sentence_word_set)
                    if len(insect) == len(group_set):
                        is_match_group = True
                        break
                if is_match_group:
                    group_to_sentence_map[idx].append(seg_sentence_text)
                    group_to_extra_map[idx].append((line['id'], line['type'], title_text))
                    is_match_cluster = True
                    break
            # 如果当前句子匹配到word_group，直接退出
            if is_match_cluster:
                break

    return group_to_sentence_map, group_to_extra_map
