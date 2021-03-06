# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 17:35
# @Author  : WangKun
# @Email   : wangkun6536@163.com

import os
import re
import pandas as pd
import unicodedata
from collections import defaultdict, namedtuple
from multiprocessing import cpu_count

from tqdm import tqdm

from nlpyutil import Logger
from nlpyutil import preprocess as preprocore
from hots import common
from hots import process
from hots import group_op
from hots import check
from hots import intervene
from hots import distance

_logger = Logger()
NUM_CPU = cpu_count()

# 过滤指定内容的文本
_FILTER_TOPIC_SENTENCE = [set(line.split()) for line in
                          open(common.FILTER_TOPIC_SENTENCE, encoding='UTF-8').readlines()]
_FILTER_TOPIC_SENTENCE = [item for item in _FILTER_TOPIC_SENTENCE if len(item) > 0]

_FILTER_TOPIC_REGEX_LIST = [line.strip()
                            for line in open(common.FILTER_TOPIC_REGEX, encoding='UTF-8').readlines() if line]
_FILTER_TOPIC_REGEX = re.compile('|'.join(_FILTER_TOPIC_REGEX_LIST))
_DELETE_TOPIC_REGEX = re.compile(r'^(回复|【图】)')
TOPIC_ACCEPT_SYMBOL = set(['+', '-', '/', '%'])
CentralGroups = namedtuple("CentralGroups", ['topic', 'groups', 'count'])


def filter_full_corpus_with_idx(data_corpus: pd.DataFrame, filtered_idx) -> pd.DataFrame:
    """
    过滤数据集
    :param corpus_path:
    :return:
    """
    filtered_corpus = data_corpus.iloc[filtered_idx]

    assert len(filtered_corpus) == len(filtered_idx)

    _logger.info("finished the filter".format(len(filtered_idx)))

    return filtered_corpus


def analyze_topic_sentence(word_groups_dict: dict,
                           full_data_dict: dict,
                           filtered_idx_dict: dict,
                           data_mark: str,
                           num_hot: int):
    """
    确定主题句
    """
    # 聚合不同渠道刷选的数据
    filtered_data_list = []
    for channel, filtered_idx in filtered_idx_dict.items():
        data_corpus = full_data_dict[channel]
        filtered_data = filter_full_corpus_with_idx(data_corpus, filtered_idx)
        filtered_data_list.append(filtered_data)
    filtered_data_pd = pd.concat(filtered_data_list).reset_index(drop=True)

    # 聚合不同渠道的word groups
    clustered_word_groups = group_op.cluster_hot_word_groups_dict(word_groups_dict)

    central_to_hot_groups, articles_to_central = find_central_sentence_with_hot_groups(filtered_data=filtered_data_pd,
                                                                                       clustered_groups=clustered_word_groups,
                                                                                       data_mark=data_mark,
                                                                                       num_hot=num_hot)

    return central_to_hot_groups, articles_to_central


def find_central_sentence_with_hot_groups(filtered_data, clustered_groups, data_mark, num_hot=50):
    """
    根据热点词组，找到相关的句子列表，并找到句子列表的中心句子
    :param corpus_path:
    :param hot_word_groups:
    :return:
    """
    topic_delete_set = intervene.get_delete_set(data_mark=data_mark)
    _logger.info("current {} intervene deleted: {}".format(data_mark, len(topic_delete_set)))
    hot_groups_str_list = []
    for cluster_info, _ in clustered_groups:
        tmp_str = "/".join([group_info["group"] for group_info in cluster_info])
        hot_groups_str_list.append(tmp_str)

    total_group_to_sentence, total_group_to_article_extra = process.preprocess_filtered_full_data(content=filtered_data,
                                                                                                  data_mark=data_mark,
                                                                                                  clustered_groups=clustered_groups)
    # 热点词组和中心句的对应：一对一
    central_to_hot_groups = []
    # 文章和热点中心句的对应：多对一
    articles_to_central = []
    hot_count = 0
    total_group_to_sentence = sorted(total_group_to_sentence.items(), key=lambda x: x[0])
    for (group_key, sentences) in tqdm(total_group_to_sentence, desc="find the hot sentence"):
        article_extras = total_group_to_article_extra[group_key]
        if hot_count >= num_hot:
            break
        central_sentence = find_central_sentence_2(sentences, article_extras)
        # 如果没有满足条件的中心句，就将词组作为中心句
        if not central_sentence:
            tmp_cluster = clustered_groups[group_key]
            central_sentence = " ".join(tmp_cluster[0][0]["group"].split("_"))
        if is_delete_topic(sentence=central_sentence,
                           topic_delete_set=topic_delete_set):
            _logger.info("drop the groups: {}, topic: {}".format(hot_groups_str_list[group_key], central_sentence))
            continue
        central_sentence = "".join(central_sentence.split())
        central_sentence = fix_topic_sentence(central_sentence)
        # 当前score为词组出现的次数
        score = clustered_groups[group_key][1]
        for extra in article_extras:
            article_id, article_type, title = extra
            articles_to_central.append([article_id, article_type, central_sentence, score])
        # central_to_hot_groups.append(CentralGroups(central_sentence, *clustered_groups[group_key]))
        central_to_hot_groups.append({"topic": central_sentence,
                                      "group_list": clustered_groups[group_key][0],
                                      "group_count": clustered_groups[group_key][1]})
        _logger.info("Hot group: {}, count: {}, central sentence: {}".format(hot_groups_str_list[group_key],
                                                                             clustered_groups[group_key][1],
                                                                             central_sentence))
        hot_count += 1

    return central_to_hot_groups, articles_to_central


def merge_central_sentence(hot_group_to_central: list, min_score=0.5):
    """
    合并相似的主题句子
    """
    merged_result = []
    central_set_list = [set(item) for item in hot_group_to_central]
    length = len(central_set_list)
    # 计算各word_group之间的相关度
    correlation = [[0.] * length for _ in range(length)]
    for idx in range(length):
        for idy in range(length):
            insect = central_set_list[idx].intersection(central_set_list[idy])
            ratio = len(insect) / len(central_set_list[idx])
            correlation[idx][idy] = max(ratio, correlation[idx][idy])
    # 根据相关度，聚合成不同的簇
    central_cluster = []
    selected = set()
    while len(selected) != length:
        choice = 0
        while choice in selected:
            choice += 1
        if choice >= length:
            break
        tmp_cluster = []
        for idx, ratio in enumerate(correlation[choice]):
            if idx not in selected and ratio > min_score:
                tmp_cluster.append(idx)
                selected.add(idx)
        central_cluster.append(tmp_cluster)
    # 对每个簇，决定其代表word_group
    for cluster in central_cluster:
        choice = None
        max_len = 0
        for idx in cluster:
            cur_len = len(hot_group_to_central[idx])
            if cur_len > max_len:
                max_len = cur_len
                choice = hot_group_to_central[idx]

        merged_result.append(choice)

    return merged_result


def is_satisfied_topic(sentence: str, min_len=4, max_len=common.MAX_TOPIC_LEN):
    """
    是否符合作为热点名的要求
    """
    sentence_str = "".join(sentence.split())
    if _FILTER_TOPIC_REGEX.search(sentence_str):
        return False

    if len(sentence_str) > max_len or len(sentence_str) < min_len:
        return False
    """
    for chr in sentence_str:
        cate = unicodedata.category(chr)
        if cate == "So":
            return False
    
    if location.is_str_location(sentence_str):
        return False
    """
    return True


def is_delete_topic(sentence, topic_delete_set: set, filter_insec_num=2, filter_insec_ratio=0.5):
    """
    是否过滤该热点
    :param sentence:
    :return:
    """
    if _DELETE_TOPIC_REGEX.search(sentence):
        return True
    sentence_words = sentence.split()
    # 没有被分词过，说明是title信息
    if len(sentence_words) == 1:
        sentence = preprocore.preprocess_text_segmentation(sentence, remove_stopwords=False)
        sentence_words = sentence.split()
    sentence_str = "".join(sentence_words)
    # 过滤包含连续重复3次以上的字符的句子
    chr_repeat_n_pattern = re.compile(r"(.)(\1){3,}")
    if chr_repeat_n_pattern.search(sentence_str):
        return True
    # 过滤在人工操作时删除的句子
    for deleted_topic in topic_delete_set:
        dist = distance.jaccard(list(deleted_topic), list(sentence_str))
        if dist >= 0.7:
            _logger.info("delete intervene topic {}".format(sentence_str))
            return True
    sentence_word_set = set(sentence.split())
    for filter_item in _FILTER_TOPIC_SENTENCE:
        insect = sentence_word_set.intersection(filter_item)
        if len(insect) / len(filter_item) > filter_insec_ratio or len(insect) >= filter_insec_num:
            return True
    """
    if location.is_str_location(sentence_str):
        _logger.info("delete the topic with pure location: {}".format(sentence_str))
        return True
    """

    return check.is_filtered_sentence(sentence_str)


def fix_topic_sentence(sentence: str):
    """
    https://graphemica.com/categories
    :param sentence:
    :return:
    """
    max_part = None
    max_len = 0
    sentence = sentence.replace(',', "，")
    for part in re.split(r'[｜|\|]', sentence):
        if len(part) > max_len:
            max_len = len(part)
            max_part = part
    sentence = max_part
    data_res = []
    for chr in sentence:
        cate = unicodedata.category(chr)
        # Symbol, Other; Symbol, Math
        if chr not in TOPIC_ACCEPT_SYMBOL and (cate == 'So' or cate == 'Sm'):
            continue
        data_res.append(chr)

    return ''.join(data_res)


def find_central_sentence_2(sentences: list, article_extras: list, title_ratio=0.3):
    """
    在句子列表中找到中心句
    :param sentences:
    :return:
    """
    sentence_count = defaultdict(int)
    title_count = defaultdict(int)
    for sentence in sentences:
        sentence_count[sentence] += 1
    for (_, _, title) in article_extras:
        if title:
            title = title.strip()
            title_count[title] += 1
    # 优先从title中获取热点名
    title_count = sorted(title_count.items(), key=lambda item: item[1], reverse=True)
    title_count = [item for item in title_count if is_satisfied_topic(item[0])]
    if title_count and title_count[0][1] / len(sentences) >= title_ratio:
        _logger.info("select title to hot: {}".format(title_count[0][0]))
        return title_count[0][0]

    sentence_count = sorted(sentence_count.items(), key=lambda item: item[1], reverse=True)
    sentence_count = [item for item in sentence_count if is_satisfied_topic(item[0])]
    if sentence_count:
        return sentence_count[0][0]
    elif title_count:
        return title_count[0][0]
    else:
        return None


if __name__ == '__main__':
    print(is_delete_topic('最高法："被执行人"未成年子女名下无正当来源大额存款可执行(注意要点)', topic_delete_set=set()))
    # print(fix_topic_sentence("真的很赞啊|啊哈|碍事法师打发斯蒂芬,你好呀"))
    # print(is_satisfied_topic("四川省甘孜藏族自治州丹巴县水利局丹巴县格宗乡格宗村太阳能光伏提灌站项目公开招标结果公告更正公告"))
