# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 16:06
# @Author  : WangKun
# @Email   : wangkun6536@163.com

import os
import time
import json
import pandas as pd
import traceback
from multiprocessing import cpu_count
from collections import defaultdict
from hots import common
from hots import count_words
from hots import corpus
from nlpyutil import Logger
from hots import group_op
from hots import topic_op, intervene

_logger = Logger()

NUM_CPU = cpu_count()


def multi_channel_pipeline(full_data_pd: pd.DataFrame,
                           name_prefix,
                           cur_hour,
                           data_mark: str,
                           col_type='type',
                           skip=[]):
    """
    整体的处理流程
    """
    word_groups_dict = {}
    filtered_idx_dict = {}
    full_data_dict = {}
    for channel_name, channel_data_pd in full_data_pd.groupby(col_type):
        _logger.info("The data of {}-{} is {}".format(data_mark, channel_name, len(channel_data_pd)))
        if channel_name in skip:
            _logger.info("{} skip the {}".format(data_mark, channel_name))
            continue
        channel_data_pd.reset_index(drop=True, inplace=True)
        processed_pd = channel_data_pd[['processed']]
        if processed_pd is None or channel_data_pd is None:
            continue
        # 统计热词
        _logger.info("{}-{} Stage 2: calculate the hot words".format(data_mark, channel_name))
        total_counter = count_words.count_words_with_corpus(processed_pd, name_prefix, data_mark)

        # 发现热点词组
        _logger.info("{}-{} Stage 3: analyze the word group".format(data_mark, channel_name))
        cur_word_groups, filtered_idx, new_groups = group_op.analyze_new_word_group(origin_corpus_data=processed_pd,
                                                                                    word_counter=total_counter,
                                                                                    channel=channel_name,
                                                                                    name_prefix=name_prefix,
                                                                                    cur_hour=cur_hour,
                                                                                    epoch=3)
        if not cur_word_groups:
            continue
        word_groups_dict[channel_name] = cur_word_groups
        filtered_idx_dict[channel_name] = filtered_idx
        full_data_dict[channel_name] = channel_data_pd

    # 分析主题句
    _logger.info("{} Stage 4: analyze the topic".format(data_mark))

    central_to_hot_groups, result_list = topic_op.analyze_topic_sentence(word_groups_dict=word_groups_dict,
                                                                         full_data_dict=full_data_dict,
                                                                         filtered_idx_dict=filtered_idx_dict,
                                                                         data_mark=data_mark,
                                                                         num_hot=30)

    # 利用人工修正的数据修正热点
    central_to_hot_groups, result_list = intervene.match_to_intervene(central_to_hot_groups=central_to_hot_groups,
                                                                      result_list=result_list,
                                                                      data_mark=data_mark)
    # 统一同名的热点
    central_to_hot_groups, result_list = merge_sim_result(central_to_hot_groups, result_list)
    # 保存数据
    save_result(result_list, central_to_hot_groups, name_prefix, cur_hour, data_mark)

    return result_list, central_to_hot_groups


def multi_pos_pipeline(data_mark, processed_path: str):
    cur_hour = time.strftime("%H", time.localtime())
    try:
        name_prefix = time.strftime("%Y%m%d", time.localtime())
        # 加载数据路径
        # processed_path = os.path.join(common.DATA_PROCESSED_PATH, name_prefix, config['watch_path'])
        full_data_df = corpus.merge_total_corpus(processed_path)
        if not isinstance(full_data_df, pd.DataFrame) or len(full_data_df) < 1:
            _logger.error("{} No processed data in {}, exit!".format(data_mark, processed_path))
            return
        multi_channel_pipeline(full_data_df, name_prefix, cur_hour, data_mark)
    except Exception as e:
        traceback.print_exc()
        _logger.error("Failed to calculate the hot of {}, error: {}".format(data_mark, e))


def merge_sim_result(central_to_hot_groups, result_list):
    """
    相同名称的合并
    :param central_to_hot_groups:
    :param result_list:
    :return:
    """
    central_count_set_map = defaultdict(set)
    for item in result_list:
        central_count_set_map[item[2]].add(item[3])
    central_count_map = {}
    for central, count_set in central_count_set_map.items():
        central_count_map[central] = sum(count_set)
        if len(count_set) > 1:
            _logger.info("{} is repeated, update the score".format(central))
    for item in result_list:
        item[3] = central_count_map[item[2]]

    return central_to_hot_groups, result_list


def save_classify_result(result, sentence_to_groups, name_prefix, cur_hour, data_mark, label_name):
    """
    保存结果
    :param result:
    :param sentence_to_groups:
    :param name_prefix:
    :param cur_hour:
    :return:
    """
    pre_name_prefix = name_prefix
    # 保存结果
    name_prefix = name_prefix + cur_hour
    result_pd = pd.DataFrame(data=result, columns=["id", "type", "hot", "score"])
    result_pd["mark"] = data_mark
    result_pd["label"] = label_name

    result_file_name = "classify_{}_topic_{}_{}.csv".format(label_name, name_prefix, data_mark)
    # 结果存储目录
    result_path = os.path.join(common.RESULT_PATH, result_file_name)
    # 结果备份目录
    result_back_path = os.path.join(common.DATA_PROCESSED_PATH, result_file_name)
    output_cols = ["id", "type", "hot", "score", "mark", "label"]
    result_pd.to_csv(result_path, index=False, header=None, encoding="utf-8", columns=output_cols)
    result_pd.to_csv(result_back_path, index=False, header=None, encoding="utf-8", columns=output_cols)

    sentence_groups_dir = os.path.join(common.SETENCE_TO_GROUPS_PATH, pre_name_prefix)
    if not os.path.exists(sentence_groups_dir):
        os.makedirs(sentence_groups_dir)
    sentence_groups_path = os.path.join(sentence_groups_dir, "{}_{}_{}.json".format(label_name, data_mark, name_prefix))

    with open(sentence_groups_path, 'w', encoding="utf-8") as f:
        json.dump(sentence_to_groups, f, ensure_ascii=False, indent=4)


def save_result(result, sentence_to_groups, name_prefix, cur_hour, data_mark):
    """
    保存结果
    :param result:
    :param sentence_to_groups:
    :param name_prefix:
    :param cur_hour:
    :return:
    """
    pre_name_prefix = name_prefix
    # 保存结果
    name_prefix = name_prefix + cur_hour
    result_pd = pd.DataFrame(data=result, columns=["id", "type", "hot", "score"])
    result_pd["mark"] = data_mark
    result_pd["label"] = ''

    for channel_name, cur_channel_pd in result_pd.groupby("type"):
        result_file_name = "{}_topic_{}_{}.csv".format(channel_name, name_prefix, data_mark)
        # 结果存储目录
        result_path = os.path.join(common.RESULT_PATH, result_file_name)
        output_cols = ["id", "type", "hot", "score", "mark", "label"]
        cur_channel_pd.to_csv(result_path, index=False, header=None, encoding="utf-8", columns=output_cols)


if __name__ == '__main__':
    multi_pos_pipeline()
