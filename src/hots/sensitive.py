# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 17:42
# @Author  : WangKun
# @Email   : wangkun6536@163.com


import os
from nlpyutil import memoize
from hots import common


@memoize(duration=None, is_log=True)
def load_sensitive_words(path=common.SENSITIVE_KEYWORDS_PATH, merge=True):
    """
    加载过滤词典
    """
    sensitive_set_list = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            cur_sensitive_set = set()
            full_path = os.path.join(root, filename)
            with open(full_path, 'r', encoding="utf-8") as f:
                cur_sensitive_set.update([word.strip() for word in f if len(word) > 1])
            sensitive_set_list.append(cur_sensitive_set)
    if merge:
        total_set = set()
        for item in sensitive_set_list:
            total_set.update(item)
        return total_set
    else:
        return sensitive_set_list


if __name__ == '__main__':
    print(load_sensitive_words())
