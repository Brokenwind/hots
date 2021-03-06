# -*- coding: utf-8 -*-
# @Time    : 10/24/2020 9:42 AM
# @Author  : WangKun
# @Email   : wangkun6536@163.com
import json
import math
from typing import Union, Generator

import pandas as pd
import pymysql
import sqlalchemy

from nlpyutil import memoize
from nlpyutil.exceptions import ValueException


class DataMysqlConfig:
    """
    数据库的配置信息
    """

    def __init__(self, host, port, database, username, password, tablename=None):
        self._host = host
        self._port = port
        self._database = database
        self._username = username
        self._password = password
        self._tablename = tablename

    @classmethod
    def from_config(cls, config) -> "DataMysqlConfig":
        if isinstance(config, str):
            return cls.from_jsonstr(config)
        elif isinstance(config, dict):
            return cls.from_dict(config)
        else:
            raise ValueException('wrong type:{} to generate instance of class DataMysqlConfig'.format(type(config)))

    @classmethod
    def from_jsonstr(cls, jsonstr) -> "DataMysqlConfig":
        """
        mysql配置参数兼容json格式字符串，但是必须包含下列字段：
        {
            "host":"192.168.201.161",
            "port":3306,
            "database":"***",
            "username":"***",
            "password":"***"
        }
        """
        json_dict = json.loads(jsonstr)
        if not isinstance(json_dict, dict):
            raise ValueError('parameter eroor with wrong database configuration')
        database_config = cls(**json_dict)
        # TODO: 这里应该对参数进行检查

        return database_config

    @classmethod
    def from_dict(cls, config_dict) -> "DataMysqlConfig":
        """
        mysql配置参数兼容json格式字符串，但是必须包含下列字段：
        {
            "host":"192.168.201.161",
            "port":3306,
            "database":"***",
            "username":"***",
            "password":"***"
        }
        """
        if not isinstance(config_dict, dict):
            raise ValueError('parameter eroor with wrong database configuration')
        database_config = cls(**config_dict)
        # TODO: 这里应该对参数进行检查

        return database_config

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    @property
    def database(self):
        return self._database

    @property
    def username(self):
        return self._username

    @property
    def password(self):
        return self._password

    @property
    def tablename(self):
        return self._tablename


@memoize(duration=None)
def get_connection_sqlalchemy(config: Union[DataMysqlConfig, str, dict]):
    '''
    获取sqlalchemy数据库链接，pandas 写入数据库时需要的connection
    mysql配置参数兼容json格式字符串，但是必须包含下列字段：
    {
        "host":"192.168.201.161",
        "port":3306,
        "database":"***",
        "username":"***",
        "password":"***"
    }
    return:
    '''

    if not isinstance(config, DataMysqlConfig):
        config = DataMysqlConfig.from_config(config)
    mysql_conn_str = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(config.username,
                                                                             config.password,
                                                                             config.host,
                                                                             config.port,
                                                                             config.database)
    connection_pool = sqlalchemy.create_engine(mysql_conn_str)

    return connection_pool


@memoize(duration=None)
def get_connection_pymysql(config: Union[DataMysqlConfig, str, dict]):
    '''
    获取pymysql数据库链接
    mysql配置参数兼容json格式字符串，但是必须包含下列字段：
    {
        "host":"192.168.201.161",
        "port":3306,
        "database":"***",
        "username":"***",
        "password":"***"
    }

    :return:
    '''
    if not isinstance(config, DataMysqlConfig):
        config = DataMysqlConfig.from_config(config)
    connection_pool = pymysql.connect(host=config.host,
                                      port=config.port,
                                      db=config.database,
                                      user=config.username,
                                      passwd=config.password,
                                      charset='utf8')

    return connection_pool


def load_db_data(mysql_config: Union[DataMysqlConfig, str],
                 tablename: str,
                 cols: list,
                 limit=None,
                 batch_size=None) -> Union[Generator, pd.DataFrame]:
    '''
    从数据库加载数据，如果设定了batch_size，则返回一个generator，否则返回一个dataframe
    :param table_name:
    :return:
    '''
    connection_pool = get_connection_pymysql(mysql_config)
    sql = "select data from " + tablename
    if limit is not None:
        sql += ' limit ' + str(limit)
    """
    if batch_size is not None:
        # 如果设定batch_sze则返回数据的迭代器，并且每次迭代获取db中batch_size大小的数据
        # 主要用于数据量过大情况，不能一次加载所有的数据，否则会OOM
        origin_data = pd.read_sql(sql, connection_pool, chunksize=batch_size)
    else:
        origin_data = pd.read_sql(sql, connection_pool)
    """
    origin_data = pd.read_sql(sql, connection_pool)
    data_dict_list = []
    for idx, item in origin_data.iterrows():
        data_dict_list.append(json.loads(item["data"]))

    return pd.DataFrame(data_dict_list)
