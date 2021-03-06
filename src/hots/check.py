# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 17:47
# @Author  : WangKun
# @Email   : wangkun6536@163.com

import numpy as np
import requests
import json
from collections import defaultdict
from nlpyutil import Logger
from hots import common

_logger = Logger()

_FILTERED_CLASSIFICATIONS = set(['娱乐', '时尚'])
_PASSED_TOPIC = ["丁真"]


def inner_cluster_similarity(sentences):
    """
    TODO: 计算一组句子的内聚程度
    :param sentences:
    :return:
    """
    sentence_count = defaultdict(int)
    sentence_len = []
    for sentence in sentences:
        sentence_count[sentence] += 1
        sentence_len.append(len(sentence))
    sentence_len = np.array(sentence_len)
    # 长度的方差
    len_var = sentence_len.std()
    # 句子的内聚程度
    sub_var = len(sentence_count) / len(sentences)


def classify_sentence(sentence: str, headers={'Content-Type': 'application/json'}):
    """
    调用远程接口进行分类
    :param sentences:
    :return:
    """
    request_data = {'text': sentence}
    label = ""
    try:
        response = requests.post(url=common.CLASSIFY_HTTP_URL,
                                 data=json.dumps(request_data),
                                 timeout=1,
                                 headers=headers)
        if response.status_code == 200:
            label = json.loads(response.text).get("data")
    except Exception as e:
        _logger.error("Error to get classification result. message:{}".format(e))

    return label


def is_filtered_sentence(sentence, except_classify=[]):
    """
    是否根据分类结果过滤
    :param sentence:
    :return:
    """
    for word in _PASSED_TOPIC:
        if word in sentence:
            return False
    label = classify_sentence(sentence)
    #if label and label in except_classify:
    #    return False
    if label and label != "正常":
        _logger.info("topic: {}, label: {}, will be dropped".format(sentence, label))
        return True
    return False


if __name__ == "__main__":
    setence_str = "阴阳两界六红二少资历金蓝娃娃菜一代黄叽盒子"
    labels = classify_sentence(sentence=setence_str)
    print(labels)
    print(is_filtered_sentence(setence_str))
