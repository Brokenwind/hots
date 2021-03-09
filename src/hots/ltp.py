# -*- coding: utf-8 -*-
# @Time    : 2020/8/28 11:21
# @Author  : WangKun
# @Email   : wangkun6536@163.com

import os
import re

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

from hots import common
from nlpyutil import Singleton


@Singleton
class Ltp(object):
    def __init__(self, model_dir=common.LTP_MODEL_PATH, exword_path=None):
        self._MODELDIR = model_dir
        self._exword_path = exword_path
        # 分词
        self._segmentor = Segmentor()
        if not self._exword_path:
            # 是否加载额外词典
            self._segmentor.load(os.path.join(self._MODELDIR, "cws.model"))
        else:
            self._segmentor.load_with_lexicon(os.path.join(self._MODELDIR, "cws.model"), self._exword_path)
        # 词性标注
        self._postagger = None
        # 依存语法
        self._parser = None
        # 命名实体识别
        self._recognizer = None
        # 语义角色识别
        self._labeller = None

    def __del__(self):
        if self._segmentor is not None:
            self._segmentor.release()
        if self._postagger is not None:
            self._postagger.release()
        if self._parser is not None:
            self._parser.release()
        if self._recognizer is not None:
            self._recognizer.release()
        if self._labeller is not None:
            self._labeller.release()

    def load_userdict(self):
        pass

    # 分词
    def ltp_segmentor(self, sentence):
        words = self._segmentor.segment(sentence)
        return words

    # 词性标注
    def ltp_postagger(self, data, is_zip=True):
        """
        词性参考: https://blog.csdn.net/leiting_imecas/article/details/68485254
        :param data:
        :return:
        """
        if self._postagger is None:
            self._postagger = Postagger()
            self._postagger.load(os.path.join(self._MODELDIR, "pos.model"))
        if isinstance(data, str):
            words = self.ltp_segmentor(data)
        else:
            words = data
        postags = self._postagger.postag(words)
        if is_zip:
            return list(zip(words, postags))
        else:
            return words, postags

    # 依存语法
    def ltp_parser(self, words, postags):
        if self._parser is None:
            self._parser = Parser()
            self._parser.load(os.path.join(self._MODELDIR, "parser.model"))
        arcs = self._parser.parse(words, postags)

        return arcs

    # 命名实体识别
    def _ltp_recognizer(self, words, postags):
        if self._recognizer is None:
            self._recognizer = NamedEntityRecognizer()
            self._recognizer.load(os.path.join(self._MODELDIR, "ner.model"))
        netags = self._recognizer.recognize(words, postags)

        return list(zip(netags, words, postags))

    # 命名实体识别
    def ltp_recognizer_str(self, data: str):
        if self._recognizer is None:
            self._recognizer = NamedEntityRecognizer()
            self._recognizer.load(os.path.join(self._MODELDIR, "ner.model"))
        words, postags = self.ltp_postagger(data, is_zip=False)
        netags = self._recognizer.recognize(words, postags)

        return list(zip(words, netags, postags))

    # 语义角色识别
    def ltp_labeller(self, sentence):
        if self._labeller is None:
            self._labeller = SementicRoleLabeller()
            self._labeller.load(os.path.join(self._MODELDIR, "pisrl.model"))
        arcs, words, postags = self.ltp_parser(sentence)
        output = []
        roles = self._labeller.label(words, postags, arcs)
        for role in roles:
            output.append([(role.index, arg.name, arg.range.start, arg.range.end) for arg in role.arguments])
        return output

    def ltp_recognizer(self, data: str):
        """
        将ner的结果进行聚合
        """
        ner_res_list = self.ltp_recognizer_str(data)
        ne_list = []
        wordtemp = ""
        for word_tag in ner_res_list:
            ne = []
            if len(word_tag) == 3 and word_tag[1] != "O":
                if re.findall("^(B|I)+", word_tag[1]):
                    # 同一实体的组合
                    wordtemp += word_tag[0]
                elif re.findall("^E+", word_tag[1]):
                    # 提取实体类型
                    wordtemp += word_tag[0]
                    tag = re.findall(r"-(.+)", word_tag[1])
                    ne.append(wordtemp)
                    ne.append(tag[0])
                    wordtemp = ""
                elif re.findall("^S+", word_tag[1]):
                    tag = re.findall(r"-(.+)", word_tag[1])
                    ne.append(word_tag[0])
                    ne.append(tag[0])
                if len(ne) != 0:
                    # 实体词汇汇总
                    ne_list.append(ne)
        return ne_list


if __name__ == '__main__':
    ltp = Ltp()
    print(common.LTP_MODEL_PATH)
    print(ltp.ltp_postagger("甘孜藏族自治州稻城亚丁风景区", is_zip=True))
    print(ltp.ltp_postagger("甘孜藏族自治州甘孜藏族自治州石渠县文化旅游局", is_zip=True))
    print(ltp.ltp_postagger("成都西南石油大学成都校区", is_zip=True))

    # ner_res_list = ltp.ltp_recognizer("甘孜藏族自治州甘孜藏族自治州石渠县文化旅游局")
    # print(ner_res_list)
    # print(ltp.ner_count(ner_res_list))
