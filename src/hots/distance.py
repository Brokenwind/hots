# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 14:36
# @Author  : WangKun
# @Email   : wangkun6536@163.com


def jaccard(word_list1: list, word_list2: list):
    '''
    Jaccard距离
    :param word_list1:
    :param word_list2:
    :return:
    '''
    word_list1 = set(word_list1)
    word_list2 = set(word_list2)
    ret1 = word_list1.intersection(word_list2)
    ret2 = word_list1.union(word_list2)
    jaccard = 1.0 * len(ret1) / len(ret2)

    return jaccard



if __name__ == '__main__':
    print(jaccard(list('我就退出家长群怎么了'), list('我就退出家长群怎么了')))
