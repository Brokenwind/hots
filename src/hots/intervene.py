# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 14:16
# @Author  : WangKun
# @Email   : wangkun6536@163.com
import datetime
from hots import distance
from sqlalchemy import Column, String, Integer, DateTime, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from hots import common
from hots import data_util
from typing import List
from nlpyutil import Logger

_logger = Logger()
base = declarative_base()


class OpinionHotRecords(base):
    __tablename__ = 'opinion_hot_records'
    id = Column(Integer(), primary_key=True)
    insert_time = Column(DateTime())
    last_update_time = Column(DateTime())
    appraise = Column(String(10))
    hot = Column(String(255))
    hot_source = Column(Integer())
    orign_appraise = Column(String(10))
    orign_hot = Column(String(255))
    orign_hot_source = Column(Integer())
    status = Column(Integer())
    orign_hot_source = Column(Integer())
    type = Column(String(15))
    hotwords = Column(String(255))

    def __repr__(self):
        return "<OpinionHotRecords(hot='%s', orign_hot=%s, status='%s')>" % (self.hot, self.orign_hot, self.status)


def query_records_with_status(type, status=None, delta_day=30, strict_modify=True) -> List[OpinionHotRecords]:
    """
    获取修改记录
    :param status: None表示修改的热点，1表示删除的热点
    """
    current_time = datetime.datetime.utcnow()
    four_weeks_ago = current_time - datetime.timedelta(days=delta_day)
    engine = data_util.get_connection_sqlalchemy(common.DB_INFO_INTERVENE)
    ##创建与数据库的会话，class,不是实例
    SessionClass = sessionmaker(bind=engine)
    # 生成session实例
    session = SessionClass()
    if status == 1:
        record_cursor = session \
            .query(OpinionHotRecords) \
            .filter(OpinionHotRecords.type == type) \
            .filter(OpinionHotRecords.status == status) \
            .filter(OpinionHotRecords.last_update_time > four_weeks_ago) \
            .all()
    else:
        if strict_modify:
            record_cursor = session \
                .query(OpinionHotRecords) \
                .filter(OpinionHotRecords.type == type) \
                .filter(or_(OpinionHotRecords.status != 1, OpinionHotRecords.status == None)) \
                .filter(OpinionHotRecords.hot != OpinionHotRecords.orign_hot) \
                .filter(OpinionHotRecords.last_update_time > four_weeks_ago) \
                .all()
        else:
            record_cursor = session \
                .query(OpinionHotRecords) \
                .filter(OpinionHotRecords.type == type) \
                .filter(or_(OpinionHotRecords.status != 1, OpinionHotRecords.status == None)) \
                .filter(OpinionHotRecords.last_update_time > four_weeks_ago) \
                .all()

    record_list = [cur for cur in record_cursor]

    return record_list


def get_modify_dict(data_mark: str) -> dict:
    """
    获取修改记录的dict
    :return:
    """
    try:
        modify_record_list = query_records_with_status(type=data_mark, status=None)
    except Exception as e:
        _logger.error("Failed to get modified data with error: {}".format(e))
        return {}
    modify_dict = {}
    for record in modify_record_list:
        modify_dict[record.orign_hot] = record.hot

    return modify_dict


def get_modify_hotwords(data_mark: str) -> dict:
    """
    获取修改记录的dict
    :return:
    """
    try:
        modify_record_list = query_records_with_status(type=data_mark, status=None, strict_modify=False)
    except Exception as e:
        _logger.error("Failed to get modified data with error: {}".format(e))
        return {}
    modify_dict = {}
    for record in modify_record_list:
        if not record.hotwords:
            continue
        modify_dict[record.hot] = record.hotwords

    return modify_dict


def get_delete_set(data_mark: str) -> set:
    """
    获取删除记录
    :return:
    """
    try:
        delete_record_list = query_records_with_status(type=data_mark, status=1)
    except Exception as e:
        _logger.error("Failed to get delete data with error: {}".format(e))
        return set()
    delete_set = set()
    for record in delete_record_list:
        delete_set.add(record.hot)
        if not record.orign_hot and record.hot != record.orign_hot:
            delete_set.add(record.orign_hot)

    return delete_set


def match_to_intervene(central_to_hot_groups: list, result_list: list, data_mark: str):
    """
    和历史数据对齐。
    根据sim_map_dict中的信息替换result_list和central_to_hot_groups中的topic信息
    :param result:
    :param sim_map_dict:
    :param central_to_hot_groups:
    :return:
    """
    modified_dict = get_modify_dict(data_mark=data_mark)
    sim_map_dict = {}
    cur_central_hots = set([cur_topic_info["topic"] for cur_topic_info in central_to_hot_groups])
    for central in cur_central_hots:
        for modified in modified_dict:
            dist = distance.jaccard(list(central), list(modified))
            if dist >= 0.7:
                sim_map_dict[central] = modified_dict[modified]
    if not sim_map_dict:
        return central_to_hot_groups, result_list
    _logger.info("intervene sim-dict: {}".format(sim_map_dict))
    for ori_topic, modified_topic in sim_map_dict.items():
        for cur_topic_info in central_to_hot_groups:
            if ori_topic == cur_topic_info["topic"]:
                cur_topic_info["topic"] = modified_topic
        for item in result_list:
            if ori_topic == item[2]:
                item[2] = modified_topic

    return central_to_hot_groups, result_list


def match_hotwords_to_intervene(result_list: list, data_mark: str):
    """
    热词和人为修改数据对齐。
    :param result:
    :param sim_map_dict:
    :param central_to_hot_groups:
    :return:
    """
    intervene_hotwords = set()
    modified_dict = get_modify_hotwords(data_mark=data_mark)
    for item in result_list:
        if item[2] in modified_dict:
            item[3] = modified_dict[item[2]]
            intervene_hotwords.add(modified_dict[item[2]])
    _logger.info("intervene hotwords: {}".format(intervene_hotwords))

    return result_list


if __name__ == '__main__':
    # query_records_with_status()
    modified = get_modify_dict(data_mark='sichuan')
    print(len(modified))
    modified = get_modify_hotwords(data_mark='sichuan')
    print(modified)
    """
    deleted = get_delete_set(data_mark='china')
    print(len(deleted))
    print(deleted)
    print('我就退出家长群怎么了' in deleted)
    """
