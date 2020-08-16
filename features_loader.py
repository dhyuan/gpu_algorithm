import pickle
from datetime import datetime
import numpy as np
import os


def __trans_source_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def restore_source_features_from_files(fid='./data/file_index_dict.dat',
                                       asf='./data/all_source_features.dat',
                                       fia='data/file_features_index_array.dat'):
    file_index_dict = {}
    all_source_features = []
    file_features_index_array = []

    persist_to_file(file_index_dict, fid)
    persist_to_file(all_source_features, asf)
    persist_to_file(file_features_index_array, fia)

    return file_index_dict, all_source_features, file_features_index_array


def read_sources_features_from_dir(features_dir, f_features_transformer=__trans_source_data_to_tuple):
    """
    从保存源特征的目录读取特征值。返回Tuple类型数据：
    第一个值是记录文件ID和文件名的对应关系的字典，
    第二个值是所有特征值作组成的一维数组 all_features；
    第三个值是一个N x 3的数组，第一列是文件ID， 第二、三列分别是该文件特征值在all_features的起始位置、结束位置。

    :param features_dir:
    :param f_features_transformer:
    :return: 返回Tuple类型数据。
    """
    time_begin = datetime.now()

    files = get_files_from_dir(features_dir)

    file_id = 0
    begin_index = 0
    total_features_numb = 0
    file_id_name_dict = {}
    all_features_array = []
    files_features_indexes_array = []
    file_feature_dict = {}  # 暂时无用，不返回。
    for file in files:
        if os.path.isdir('%s/%s' % (features_dir, file)) or not file.endswith('.txt') or file.startswith(
                '.') or os.path.getsize('%s/%s' % (features_dir, file)) == 0:
            print('忽略空文件: %s' % file)
            continue

        source_file = '%s/%s' % (features_dir, file)
        source_file_features = text_read(source_file)
        print('src-file %d  %s %d' % (file_id, source_file, len(source_file_features)))

        # 用序号做ID、记录ID和文件名对应关系。
        file_id_name_dict[file_id] = file

        # 解析文件中的features
        source_file_features = list(map(lambda d: f_features_transformer(d), source_file_features))

        # 拼接此文件的features到特征合集。
        all_features_array.extend(source_file_features)

        # 记录每个文件对应的特征集
        source_features_array = np.array(source_file_features, dtype=np.float32)
        file_feature_dict[file] = source_features_array

        source_features_numb = len(source_file_features)
        total_features_numb += source_features_numb
        files_features_indexes_array.append([file_id, begin_index, total_features_numb])
        begin_index = total_features_numb

        file_id += 1

    all_features = np.asarray(all_features_array, dtype=np.float32)
    file_features_indexes = np.asarray(files_features_indexes_array, dtype=np.int32)

    time_end = datetime.now()
    print('读取%d个源特征文件并解析用时：%s' % (len(files), (time_end - time_begin)))
    return file_id_name_dict, all_features, file_features_indexes


def __trans_test_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def read_test_features_from_file(testfile, fun_features_str2tuple=__trans_test_data_to_tuple):
    list_of_features = text_read(testfile)
    return np.array(list(map(lambda d: fun_features_str2tuple(d), list_of_features)), dtype=np.float)


def text_read(f):
    try:
        lines = open(f, 'r').readlines()
        return lines
    except Exception as e:
        print(e)
        print('ERROR, 结果文件不存在！')


def trans_source_data_to_list(data):
    ds = data.split()
    return [float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])]


def trans_test_data_to_list(data):
    ds = data.split()
    return [float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])]


def get_files_from_dir(src_dir):
    files = os.listdir(src_dir)
    return files


def persist_to_file(data, file_name):
    f = open(file_name, 'wb')
    pickle.dump(data, f)
    f.close()


def restore_from_file(file_name):
    f = open(file_name, 'rb')
    d = pickle.load(f)
    f.close()
    return d
