# -*- coding: utf-8 -*-
# @Time    : 2020/9/1 18:00
# @Author  : WangKun
# @Email   : wangkun6536@163.com

import os
import pickle
import sys
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from hots import common
from nlpyutil import preprocess as preprocore
from hots import corpus
from nlpyutil import Logger
from .ltp import Ltp

_mds = set()

DEFAULT_CSV_COLS = 5
_logger = Logger()
_ltp = Ltp(exword_path=common.SELF_USER_WV_DICT)
_HOT_FILTER_WORDS = set([word.strip() for word in open(common.FILTER_HOT_WORDS_PATH, encoding='UTF-8').readlines()])
_STOPWORDS = preprocore.get_stopwords()


def word_count_and_idf(corpus, idf=False):
    """
    词频统计 和 idf值计算
    :param corpus: [文章1 空白分割, 文章2 空白分割]
    :return:
      word_list: 词典列表
      word_count_list: 词统计列表
      idf_list: idfs列表
    """
    if not corpus:
        return None, None, None
    cv = CountVectorizer(stop_words=_STOPWORDS)
    cv_fit = cv.fit_transform(corpus)
    # ['bird', 'cat', 'dog', 'fish'] 列表形式呈现文章生成的词典
    word_list = cv.get_feature_names()
    # 词频统计
    word_count_list = cv_fit.sum(axis=0).tolist()
    word_count_list = word_count_list[0]
    assert len(word_list) == len(word_count_list)
    # analyzer='word',token_pattern=u"(?u)\\b\\w+\\b"
    idf_list = []
    if idf:
        transformer = TfidfTransformer()
        tfidf = transformer.fit(cv_fit)
        # 计算全局的tfidf值
        idf_list = tfidf.idf_
        assert len(idf_list) == len(word_count_list)

    return word_list, word_count_list, idf_list


def count_and_filter_single(partial_corpus):
    word_list, word_count_list, idf_list = word_count_and_idf(partial_corpus, idf=False)
    if not word_list or not word_count_list:
        return Counter()
    # word_count_list = np.array(word_count_list)
    # idf_list = np.array(idf_list)
    # tfidf_list = word_count_list * idf_list

    # word_statistics_list = list(zip(word_list, word_count_list, idf_list, tfidf_list))
    word_counter = Counter(dict(zip(word_list, word_count_list)))

    return word_counter


def count_words_with_corpus(corpus_data, name_prefix, data_mark, allow_tags=['n', 'j', 'i'], deny_tags=['nt', 'nd']):
    """
    在处理好的数据集上进行词频统计
    :param filepath:
    :param count_base: 词的最低频次
    :param idf_base: 最低idf限制
    :param tfidf_base: 最低tfidf限制
    :param allow_tags: 统计指定词性的词，如果为空则表示所有词性
    :return:
    """
    corpus_data, _ = corpus.process_origin_corpus_data(corpus_data, split=False,
                                                       level=corpus.CorpusItemLevel.Article)

    _logger.info("start counting with multi thread")
    num_task = max(1, common.NUM_CPU)
    partial_len = int(np.ceil(len(corpus_data) / num_task))
    if partial_len == 0:
        partial_len = len(corpus_data)
        num_task = 1
    partial_results = Parallel(n_jobs=num_task, backend="multiprocessing")(
        delayed(count_and_filter_single)(corpus_data[idx:idx + partial_len])
        for idx in range(0, len(corpus_data), partial_len))
    total_counter = Counter()

    for counter in partial_results:
        total_counter.update(counter)

    # 根据条件进行过滤
    _logger.info("start filtering the with postag and count")
    # 根据词性过滤
    for word in list(total_counter.keys()):
        if total_counter[word] < 10:
            del total_counter[word]
            continue
        if word in _HOT_FILTER_WORDS:
            del total_counter[word]
            continue
        word_tags = _ltp.ltp_postagger(word)
        if len(word_tags) == 1:
            word_tag = word_tags[0]
            # 如果当前词的词性属于deny_tags中，则直接跳过
            deny = any(word_tag[1].startswith(tag) for tag in deny_tags)
            if deny:
                del total_counter[word]
                continue
            allow = any(word_tag[1].startswith(tag) for tag in allow_tags)
            if not allow:
                del total_counter[word]
    """
    word_statistics_list_path = os.path.join(common.HOT_WORDS_PATH,
                                             "{}_word_statistics_{}.pkl".format(data_mark, name_prefix))
    
    _logger.info("dump to pickle file")
    with open(word_statistics_list_path, 'wb') as f:
        pickle.dump(total_counter, f)
    _logger.info("finished the dump")
    """
    return total_counter


def load_word_statistics(name_prefix) -> Counter:
    """
    :param name_prefix:
    :return: 返回词统计结果列表，每个元素：('词', 个数, idf值, 全局的tf-idf值)
    """
    word_counter_path = os.path.join(common.HOT_WORDS_PATH, "word_statistics_{}.pkl".format(name_prefix))

    with open(word_counter_path, 'rb') as f:
        word_counter = pickle.load(f)

    return word_counter


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("parameter error")
    name_prefix = sys.argv[1]
    filepath = os.path.join(common.DATA_PROCESSED_PATH, name_prefix + ".corpus")
    count_words_with_corpus(filepath)

"""
corpus = ["我 来到 北京 清华大学",
          "他 来到 了 网易 杭研 大厦",
          "小明 硕士 毕业 与 中国 科学院",
          "我 爱 北京 天安门"]
word_list, word_count_list, idf_list = word_count_and_idf(corpus)
print(word_list)
print(word_count_list)
print(idf_list)
"""
