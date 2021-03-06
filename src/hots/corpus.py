# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 17:10
# @Author  : WangKun
# @Email   : wangkun6536@163.com

import os
import pickle
from enum import Enum
import itertools
import numpy as np
import pandas as pd
from nlpyutil import Logger, STOPWORDS
from newords import NUM_CPU, SENTENCE_JOINT_CHR
from tqdm import tqdm
from joblib import Parallel, delayed

_logger = Logger()


class CorpusItemLevel(Enum):
    Sentence = "Sentence"
    Article = "Article"


def load_processed_data(filepath,
                        split=False,
                        data_idx=-1,
                        min_sentence_len=3,
                        level=CorpusItemLevel.Article,
                        sentence_joint_chr='，') -> tuple:
    """
    :param filepath 文件路径
    :param split 是否对词进行切割
    :param article 是否区分文章
    加载文件内容
    文件中每条记录的格式：一条记录中的column中包含一篇文章，
                        文章中每个句子由sentence_joint_chr分开，
                        文章中每个词以空格分开

    如果split==True， 则先将文章切成句子，然后句子再切成词, return: [[句子1的词列表，句子2的词列表], [句子1的词列表，句子2的词列表]...]
    如果split==False, 直接返回column指定内容, return: [[句子1，句子2], [句子1，句子2]...]
    """
    result_data = []
    sent_article_map = []
    content = pd.read_table(filepath,
                            sep=',',
                            encoding='utf-8',
                            dtype=np.str,
                            error_bad_lines=False,
                            header=None)
    content.fillna(value="", inplace=True)
    _, filename = os.path.split(filepath)
    data_idx = content.shape[1] - 1 if data_idx == -1 else data_idx
    for idx, line in tqdm(content.iterrows(), "loading {}".format(filename)):
        item = line[data_idx]
        if level == CorpusItemLevel.Article:
            if split:
                result_data.append(item.replace(sentence_joint_chr, ' ').split())
            else:
                result_data.append(item.replace(sentence_joint_chr, ' '))
        else:
            for sentence in item.split(sentence_joint_chr):
                if not sentence or sentence == '\n' or sentence == '\r\n':
                    continue
                if len(sentence) < min_sentence_len:
                    continue
                if split:
                    result_data.append(sentence.split())
                else:
                    result_data.append(sentence)
                sent_article_map.append(idx)

    if level == CorpusItemLevel.Sentence:
        assert len(sent_article_map) == len(result_data)
        return result_data, sent_article_map
    else:
        return result_data, None


def load_processed_files_to_corpus(filepath, split=False, level=CorpusItemLevel.Article):
    """
    加载用于训练LDA的数据集
    :param filepath:
    :return:
    """
    if os.path.isdir(filepath):
        file_list = [os.path.join(filepath, name) for name in os.listdir(filepath)
                     if os.path.isfile(os.path.join(filepath, name))]
    elif os.path.isfile(filepath):
        file_list = [filepath]
    else:
        raise ValueError("the filepath: {} is not valid".format(filepath))
    corpus = []
    for filename in tqdm(file_list, desc="loading file"):
        filedata, _ = load_processed_data(filename, split=split, level=level)
        corpus.extend(filedata)

    return corpus


def load_corpus(filepath, split=False, level=CorpusItemLevel.Article):
    """
    corpus的格式:
    1. 每个句子为一行
    2. 文章之间由一个换行隔开

    :param filepath: corpus的路径
    :param level:    CorpusItemLevel.Article/CorpusItemLevel.Sentence
    :return:
    """
    if not os.path.exists(filepath):
        raise ValueError("the filepath: {} is not valid".format(filepath))
    return load_processed_data(filepath, split=split, level=level)


def load_for_merge(full_filepath, is_remove=False):
    _, filename = os.path.split(full_filepath)
    if not os.path.isfile(full_filepath):
        return None, filename
    try:
        data_pd = pd.read_csv(full_filepath)
    except Exception as e:
        _logger.error("Failed to open the file:{}, error:{}".format(filename, e))
        if is_remove:
            os.remove(full_filepath)
        return None, filename

    if is_remove:
        os.remove(full_filepath)

    return data_pd


def merge_total_corpus(filepath) -> pd.DataFrame:
    if not os.path.isdir(filepath):
        _logger.error("the filepath: {} is not valid".format(filepath))
        return None, None

    file_paths = []
    for inputfile in os.listdir(filepath):
        full_filepath = os.path.join(filepath, inputfile)
        if not os.path.isfile(full_filepath):
            continue
        file_paths.append(full_filepath)
    if not file_paths:
        _logger.warn("no data to be loaded, please check it")
        return None, None

    _logger.info("start loading files with multi-processing")
    num_task = NUM_CPU
    partial_result_list = Parallel(n_jobs=num_task, backend="multiprocessing")(
        delayed(load_for_merge)(path) for path in file_paths)

    total_data = pd.concat(partial_result_list, axis=0, ignore_index=True)

    return total_data


def process_origin_corpus_data_single(content: pd.DataFrame,
                                      start_idx: int,
                                      split=False,
                                      data_idx=-1,
                                      min_sentence_len=3,
                                      level=CorpusItemLevel.Article):
    """
    对数据集进行重新整理，以文章还是句子为最小单位
    :param content:
    :param start_idx:
    :param split:
    :param data_idx:
    :param min_sentence_len:
    :param level:
    :return:
    """
    sentence_joint_chr = SENTENCE_JOINT_CHR
    result_data = []
    # 句子和其文章对应对应， sent_article_map[i] = j，表示第i个句子属于第j篇文章
    sent_article_map = []
    content.fillna(value="", inplace=True)
    data_idx = content.shape[1] - 1 if data_idx == -1 else data_idx
    for idx, line in tqdm(content.iterrows(), "processing corpus"):
        item = line[data_idx]
        if level == CorpusItemLevel.Article:
            if split:
                result_data.append(item.replace(sentence_joint_chr, ' ').split())
            else:
                result_data.append(item.replace(sentence_joint_chr, ' '))
        else:
            for sentence in item.split(sentence_joint_chr):
                if not sentence or sentence == '\n' or sentence == '\r\n':
                    continue
                if len(sentence) < min_sentence_len:
                    continue
                if split:
                    result_data.append(sentence.split())
                else:
                    result_data.append(sentence)
                # idx并不一定从0开始，不需要加上偏移量start_idx
                sent_article_map.append(idx)

    if level == CorpusItemLevel.Sentence:
        assert len(sent_article_map) == len(result_data)
        return result_data, sent_article_map
    else:
        return result_data, None


def process_origin_corpus_data(content: pd.DataFrame,
                               split=False,
                               data_idx=-1,
                               min_sentence_len=3,
                               level=CorpusItemLevel.Article):
    num_task = NUM_CPU
    partial_len = int(np.ceil(len(content) / num_task))
    if partial_len == 0:
        partial_len = len(content)
        num_task = 1
    partial_results = Parallel(n_jobs=num_task, backend="multiprocessing")(
        delayed(process_origin_corpus_data_single)(content.iloc[idx:idx + partial_len],
                                                   idx,
                                                   split,
                                                   data_idx,
                                                   min_sentence_len,
                                                   level)
        for idx in range(0, len(content), partial_len))
    _logger.info("merge the result of multi-processing")
    # 合并结果
    result_data_list = [partial[0] for partial in partial_results]
    result_data = list(itertools.chain(*result_data_list))
    if level == CorpusItemLevel.Sentence:
        sent_article_map_list = [partial[1] for partial in partial_results]
        sent_article_map = list(itertools.chain(*sent_article_map_list))
        assert len(sent_article_map) == len(result_data)
    else:
        sent_article_map = None

    return result_data, sent_article_map


def filter_corpus_with_anchor_words(origin_corpus,
                                    name_prefix,
                                    anchor_words=[],
                                    remove_stopwords=True,
                                    is_save_idx=False):
    """
    过滤数据集
    :param corpus_path:
    :param anchor_words:
    :param remove_stopwords:
    :return:
    """
    data_corpus, sent_article_map = process_origin_corpus_data(origin_corpus,
                                                               split=True,
                                                               level=CorpusItemLevel.Sentence)
    sentence_count = 0
    anchor_words_set = set(anchor_words)
    filtered_corpus = []
    filtered_idx = set()
    for idx, word_list in tqdm(enumerate(data_corpus), desc="filter corpus with anchor words"):
        if remove_stopwords:
            word_list = [word for word in word_list if word not in STOPWORDS]
        if len(word_list) < 2:
            continue
        if anchor_words:
            insection = anchor_words_set.intersection(set(word_list))
            if not insection:
                continue
        filtered_idx.add(sent_article_map[idx])
        filtered_corpus.append(word_list)
        sentence_count += 1
    _logger.info("About {} sentences that contain the anchor words.".format(sentence_count))

    filtered_idx = sorted(list(filtered_idx))

    return filtered_corpus, filtered_idx


if __name__ == '__main__':
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_path, 'data')
    data = merge_total_corpus(data_path)
