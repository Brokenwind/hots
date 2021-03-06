# -*- coding: utf-8 -*-
# @Time    : 26/8/2020 10:53 AM
# @Author  : WangKun
# @Email   : wangkun6536@163.com

import os
import yaml
from enum import Enum
from multiprocessing import cpu_count

TOTAL_NUM_CPU = cpu_count()


class Env(Enum):
    Online = "online"
    Dev = "dev"


# 并行处理时的线程数
PARALLEL_THREAD_NUM = 4
# 当前目录
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(CUR_PATH, '../../'))
# 配置文件路径
CONFIG_PATH = os.path.join(PROJECT_ROOT_PATH, 'configs')
_config = {}
NUM_CPU = (int)(0.2 * TOTAL_NUM_CPU)
# 是否备份处理后的数据
IS_BACKUP_PROCESSED = _config.get("is_backup_processed", False)
# 聚合group的最小相似分数
CLUSTER_MIN_SCORE = _config.get("cluster_min_score", 0.65)
# 文章最大长度
ARTICLE_MAX_LENGTH = _config.get("article_max_length", 150)
# 利用全部数据时的浮动字符数
FULL_FLOAT_CHR_NUM = _config.get("full_float_chr_num", 100)
# topic的最大长度
MAX_TOPIC_LEN = _config.get("max_topic_len", 35)
# 认为是一个group的最小count数量
MIN_GROUP_COUNT = _config.get("min_group_count", 2)
# data所在目录
DATA_PATH = _config.get("data_path", os.path.join(PROJECT_ROOT_PATH, 'data'))
# 其它依赖库的地址
LIBS_PATH = os.path.join(PROJECT_ROOT_PATH, 'libs')
# configs 所在目录
CONFIGS_PATH = os.path.join(PROJECT_ROOT_PATH, 'configs')
# word2vec
WORD2VEC_PATH = os.path.join(DATA_PATH, 'word2vec/word2vec_wx')
# 新词发现相关数据文件
NEW_WORDS_DATA_PATH = os.path.join(DATA_PATH, 'new_words')
# 词组文件的位置
PHRASE_DATA_PATH = os.path.join(DATA_PATH, 'phrase')
# ltp模型文件地址
LTP_MODEL_PATH = os.path.join(DATA_PATH, 'ltp_models')
# 预训练的词嵌入目录
WORD2VEC_PATH = os.path.join(DATA_PATH, 'word2vec/word2vec_wx')
# 数据集存放的位置
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
# 原始数据集的位置
DATASET_ORIGIN_PATH = os.path.join(DATASET_PATH, 'dataset_origin')
# 停用词文件
STOPWORDS_FILE_PATH = os.path.join(DATA_PATH, "stopwords.txt")
# 生僻词词典
RAREWORDS_FILE_PATH = os.path.join(DATA_PATH, "rarewords.txt")
# 分词添加的字典
SELF_USER_DICT = os.path.join(DATA_PATH, "userdict.txt")
# 分词添加的字典
SELF_USER_WV_DICT = os.path.join(DATA_PATH, "dict_wv.txt")
# 地理位置词典
LOCATION_DICT = os.path.join(DATA_PATH, "dict_location.txt")
# 重点关注词汇
MARJOR_DICT = os.path.join(DATA_PATH, "dict_major.txt")
# 词/字 列表文件
VOCABULARY_FILE_DIR = os.path.join(DATA_PATH, "vocabulary")
# 预训练模型位置
MODEL_PRETRAINED_PATH = os.path.join(DATA_PATH, 'model_pretrained')
# count ngram程序的地址
BIN_COUNT_NGRAM_PATH = os.path.join(LIBS_PATH, "count_ngrams")
# 词频数据文件
FREQ_DICT_FILE_PATH = os.path.join(DATA_PATH, "freq_dict.json")
# 热词的屏蔽词文件
FILTER_HOT_WORDS_PATH = os.path.join(DATA_PATH, "filter_hot_words.txt")
# 新词发现时，需要过滤的词汇
FILTER_WORD_GROUPS_PATH = os.path.join(DATA_PATH, "filter_word_groups.txt")
# 数据预处理时，需要过滤的句子
FILTER_NOISE_SENTENCE_PATH = os.path.join(DATA_PATH, "filter_noise.txt")
# 数据预处理时，需要过滤的句子的正则表达式
FILTER_NOISE_REGEX_PATH = os.path.join(DATA_PATH, "filter_noise_regex.txt")
# 过滤备选中心句的某些场景
FILTER_TOPIC_SENTENCE = os.path.join(DATA_PATH, "filter_topic_sentence.txt")
# 过滤备选中心句的某些正则场景
FILTER_TOPIC_REGEX = os.path.join(DATA_PATH, "filter_topic_regex.txt")
# 过滤包含其中句子的文章
FILTER_ARTICLE_SENTENCE = os.path.join(DATA_PATH, "filter_article_sentence.txt")
# 在聚合goups的时候，需要排除的词
FILTER_WORD_IN_CLUSTER = os.path.join(DATA_PATH, "filter_word_in_cluster.txt")
# 需要排除的机构
FILTER_ORG = os.path.join(DATA_PATH, "filter_org.txt")
# 文件处理后，句子的分隔符
SENTENCE_JOINT_CHR = '，'
# 位置信息文件
LOCATION_INFO_FILE_PATH = os.path.join(DATA_PATH, 'pca.csv')
# 原始词库文件位置
ORIGIN_LEXICON_PATH = os.path.join(DATA_PATH, 'origin_lexicon')
# 处理好的词库位置
LEXICON_PATH = os.path.join(DATA_PATH, 'lexicon')
# 需要过滤的词库位置
TOPIC_FILTER_LEXICON_PATH = os.path.join(DATA_PATH, 'topic_filter_lexicon')
# 分类接口
CLASSIFY_HTTP_URL = _config.get("classify_http_url", "")
# 模型名或者位置
MODEL_NAME_OR_PATH = "voidful/albert_chinese_tiny"
# 人工干预数据库信息
DB_INFO_INTERVENE = _config.get("db_info_intervene", {})
# 敏感词库位置
SENSITIVE_KEYWORDS_PATH = os.path.join(DATA_PATH, 'sensitive')

