import json
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import sys

def list_directories(pathname, predicate=lambda x: True):
    """列出目录下的所有子目录
    
    参数:
        pathname: 目录路径
        predicate: 过滤函数
        
    返回:
        符合条件的子目录列表
    """
    # 如果目录不存在，创建它
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print(f"创建目录: {pathname}")
        return []  # 新创建的目录为空
        
    return [join(pathname, f) for f in listdir(pathname) 
            if not isfile(join(pathname, f)) and predicate(f)]

def list_files(pathname, predicate=None):
    if predicate is None:
        predicate = lambda x: True
    return [join(pathname, f) for f in listdir(pathname) if isfile(join(pathname, f)) and predicate(f)]

def list_files_with_predicate(pathname, predicate):
    return [join(pathname, f) for f in listdir(pathname) if predicate(pathname, f)]

def mkdir_with_timestap(pathname):
    datetimetext = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dir_name = "{}_{}".format(pathname, datetimetext)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def save_json(filename, data):
    with open(filename, 'w') as data_file:
        json.dump(data, data_file)


def load_json(filename):
    data = None
    with open(filename) as data_file:    
        data = json.load(data_file)
    assert data is not None
    return data
