# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 17:39
# @Author  : WangKun
# @Email   : wangkun6536@163.com

import itertools
import math
import os
import re
from collections import Counter
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from newords import find_new_words_with_anchor_words
from nlpyutil import Logger
from tqdm import tqdm

from . import common
from . import corpus

_logger = Logger()

NUM_CPU = cpu_count()

_FILTER_WORD_IN_CLUSTER = [line.strip()
                           for line in open(common.FILTER_WORD_IN_CLUSTER, encoding='UTF-8').readlines() if line]

_FILTER_WORD_IN_CLUSTER_REGEX = re.compile('|'.join(_FILTER_WORD_IN_CLUSTER))
_FILTER_WORD_IN_CLUSTER_SET = set(_FILTER_WORD_IN_CLUSTER)


def merge_word_group_single(corpus: list, word_groups: list, delimiter="_"):
    """
    将corpus中出现的词组进行合并
    :param corpus:
    :param word_groups:
    :return:
    """
    total_word_group_set = set()
    word_group_set_list = []
    for word_group in word_groups:
        word_group_set_list.append(set(word_group))
        total_word_group_set.update(word_group)
    for sentence_idx, word_list in tqdm(enumerate(corpus), desc="merge groups"):
        if len(word_list) < 2:
            continue
        word_set = set()
        splitted_word_list = []
        for splitted_word in word_list:
            tmp_splited = splitted_word.split(delimiter)
            splitted_word_list.append(tmp_splited)
            word_set.update(tmp_splited)
        insect = total_word_group_set.intersection(word_set)
        # 通过判断是否有交集，如果没有交集，就不用后边的计算，加速计算速度
        if not insect:
            continue
        for group_idx, word_group in enumerate(word_groups):
            if not word_group_set_list[group_idx].intersection(word_set):
                continue
            new_word_list = []
            idx = 0
            while idx < len(splitted_word_list):
                splitted_word = splitted_word_list[idx]
                next_word = splitted_word_list[idx + 1] if idx + 1 < len(splitted_word_list) else None
                if next_word \
                        and splitted_word[-1] == word_group[0] \
                        and next_word[0] == word_group[1]:
                    new_word_list.append(splitted_word + next_word)
                    idx += 2
                else:
                    new_word_list.append(splitted_word)
                    idx += 1
            splitted_word_list = new_word_list
        corpus[sentence_idx] = [delimiter.join(splitted_word) for splitted_word in splitted_word_list]

    return corpus


def merge_word_group(corpus: list, word_groups: list, delimiter="_"):
    """
    合并
    :param corpus:
    :param word_groups:
    :return:
    """
    if common.ENVIRONMENT == common.Env.Dev.value:
        num_task = 1
    else:
        num_task = max(1, common.NUM_CPU)
    partial_len = int(np.ceil(len(corpus) / num_task))
    if partial_len == 0:
        partial_len = len(corpus)
        num_task = 1
    results = Parallel(n_jobs=num_task, backend="multiprocessing")(
        delayed(merge_word_group_single)(corpus[idx:idx + partial_len], word_groups, delimiter)
        for idx in range(0, len(corpus), partial_len))
    corpus = list(itertools.chain(*results))

    return corpus


def cluster_hot_word_groups_dict(word_groups_dict: dict, min_score=common.CLUSTER_MIN_SCORE):
    """
    热点会重复，需要进行聚合
    :param hot_word_groups:
    :return:
    """
    total_word_groups = []
    for channel, word_groups in word_groups_dict.items():
        total_word_groups.extend(word_groups)
    _logger.info("There are {} word groups".format(len(total_word_groups)))
    clustered_word_groups = cluster_hot_word_groups(total_word_groups, min_score=min_score)
    _logger.info("Finished the clustering groups")

    return clustered_word_groups


def cluster_hot_word_groups(word_groups_result, min_score=common.CLUSTER_MIN_SCORE, improve_ratio=1.075):
    """
    热点会重复，需要进行聚合
    :param hot_word_groups:
    :return:
    """
    groups_word_set_list = [set(item[0].split("_")) for item in word_groups_result]
    groups_str_list = ["".join(item[0].split("_")) for item in word_groups_result]
    # groups_str_list = [_FILTER_WORD_IN_CLUSTER_REGEX.sub("", item) for item in groups_str_list]
    groups_set_list = [set(item) for item in groups_str_list]
    groups_len_list = [len(item) for item in groups_set_list]
    groups_avg_len = sum(groups_len_list) / len(groups_len_list)
    length = len(groups_set_list)
    # 计算各word_group之间的相关度
    correlation = [[0.] * length for _ in range(length)]
    for idx in range(length):
        idx_len = len(groups_set_list[idx])
        if idx_len < 5:
            idx_len = groups_avg_len
        for idy in range(length):
            if idx == idy:
                ratio = 1.0
            else:
                insect = groups_set_list[idx].intersection(groups_set_list[idy])
                ratio = len(insect) / idx_len
            correlation[idx][idy] = max(ratio, correlation[idx][idy])
    # 根据相关度，聚合成不同的簇
    group_cluster = []
    selected = set()
    while len(selected) != length:
        choice = 0
        while choice in selected:
            choice += 1
        if choice >= length:
            break
        cur_cluster = []
        cluster_count = 0
        for idx, ratio in enumerate(correlation[choice]):
            insect = groups_word_set_list[idx].intersection(_FILTER_WORD_IN_CLUSTER_SET)
            insect_chr_len = 0
            for item in insect:
                insect_chr_len += len(set(item))
            insect_ration = insect_chr_len / groups_avg_len
            # 当相交的字符比例较大时，需要提高基数
            if insect_ration > 0.4:
                cur_min_score = min(min_score * math.pow(improve_ratio + math.pow(insect_ration, 2), len(insect)), 1)
            else:
                if insect:
                    cur_min_score = min(min_score * math.pow(improve_ratio, len(insect)), 1)
                else:
                    cur_min_score = min_score
            if idx not in selected and ratio >= cur_min_score:
                cur_item = word_groups_result[idx]
                cur_cluster.append({"group": cur_item[0], "score": cur_item[1], "count": cur_item[2]})
                cluster_count += cur_item[2]
                selected.add(idx)
        # 包含簇中的词组，簇的大小
        item_info = (cur_cluster, cluster_count)
        group_cluster.append(item_info)

    group_cluster = sorted(group_cluster, key=lambda x: x[1], reverse=True)

    return group_cluster


def filter_word_group(result, delimiter="_", min_words=2, min_word_len=6):
    """
    过滤词组
    """
    filetered_result = []
    for item in result:
        words = item[0].split(delimiter)
        len_word = len(item[0]) - len(words) + 1
        if len(words) < min_words or len_word < min_word_len:
            continue
        filetered_result.append(item)
    if not filetered_result:
        return result[:10]

    return filetered_result


def analyze_new_word_group(origin_corpus_data,
                           word_counter: Counter,
                           channel: str,
                           name_prefix,
                           cur_hour,
                           top_n=500,
                           epoch=3,
                           delimiter="_",
                           min_words=2,
                           min_word_len=6,
                           ):
    """
    根据热词，分析热词的词组
    """
    word_counter = word_counter.most_common(top_n)
    selected_words = [item[0] for item in word_counter]
    _logger.info("selected words: {}".format(selected_words[0:top_n]))
    # 根据跟定的锚定词过滤文章
    filtered_corpus_data, filtered_idx = corpus.filter_corpus_with_anchor_words(origin_corpus_data,
                                                                                name_prefix=name_prefix,
                                                                                anchor_words=selected_words)

    _logger.info("start {} epoch searching".format(0))
    epoch_result_list = []
    pre_result = None
    result = find_new_words_with_anchor_words(filtered_corpus_data,
                                              load_previous=False,
                                              previous_model_path="",
                                              desc=name_prefix)
    epoch_result_list.append(result)
    _logger.info("finished {} epoch searching".format(0))
    for idx in range(1, epoch):
        _logger.info("start {} epoch searching".format(idx))
        new_word_groups = []
        if not result:
            _logger.warn("there is not enough data to calculate {} word-group".format(idx))
            result = pre_result
            break
        for item in result[0:top_n * 2]:
            new_word_groups.append(item[0].split(delimiter))
        filtered_corpus_data = merge_word_group(filtered_corpus_data, new_word_groups)
        pre_result = result
        result = find_new_words_with_anchor_words(filtered_corpus_data,
                                                  load_previous=False,
                                                  previous_model_path="",
                                                  desc=name_prefix)
        epoch_result_list.append(result)
        _logger.info("finished {} epoch searching".format(idx))
    if not result:
        _logger.warn("not found the groups")
        return None, None, None
    _logger.info("filtering the result")
    filetered_result = filter_word_group(result, delimiter=delimiter, min_words=min_words, min_word_len=min_word_len)
    _logger.info("writing result to file")

    # 保存词组结果
    new_word_path = os.path.join(common.WORD_GROUPS_PATH,
                                 "{}{}_{}.csv".format(name_prefix, cur_hour, channel))
    result_pd = pd.DataFrame(data=filetered_result, columns=["word", "score", "count"])
    result_pd.to_csv(new_word_path, index=False, encoding="utf-8")

    return filetered_result[0:1000], filtered_idx, epoch_result_list


if __name__ == '__main__':
    word_groups_str = "我国_疫情_防控/疫情_防控_经济社会_发展/疫情_防控_视频会议/全市_疫情_防控/介绍_加强_冬季_疫情_防控_深入开展/社区_开展_疫情_防控_应急_演练/秋冬季_疫情_防控/统筹_推进_疫情_防控/介绍_加强_冬季_疫情_防控_情况/部门_介绍_加强_冬季_疫情_防控/进口_冷链_食品_疫情_防控/疫情_防控_常态化/开展_疫情_防控/部门_介绍_加强_冬季_疫情_防控_情况/做好_疫情_防控/秋冬季_疫情_防控/疫情_防控_这根/疫情_防控_应急/感染_肺炎_疫情_防控/疫情_防控_常态化/疫情_防控_经济社会_发展/疫情_防控_指挥部/疫情_防控_成果/疫情_防控_期间"
    # word_groups_str = "四川_甘孜州_藏族人/位于_四川省_甘孜藏族自治州/四川省_甘孜藏族自治州_巴塘县/概况_四川省_甘孜藏族自治州/四川省_甘孜藏族自治州_道孚县/四川省_甘孜藏族自治州"
    word_groups = word_groups_str.split("/")
    word_groups = list(zip(word_groups, [0.6] * len(word_groups), [2] * len(word_groups)))
    print(word_groups)
    res = cluster_hot_word_groups(word_groups_result=word_groups)
    print(len(res))
