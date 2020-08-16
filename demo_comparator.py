from datetime import datetime


from features_comparator import copy_all_sources_features_to_gpu, compare_by_gpu
from features_comparator import copy_test_features_to_gpu, calculate_result
from features_loader import read_sources_features_from_dir, read_test_features_from_file, get_files_from_dir


def demo_compare(test_dir, sources_dir, value=0.05):
    # Step 1: 从source features目录读取特征值。
    file_index_dict, all_source_features, file_features_index_array = read_sources_features_from_dir(sources_dir)

    # Step 2: 将所有源特征合集导入GPU
    d_all_sources_features_in_gpu = copy_all_sources_features_to_gpu(all_source_features)

    test_files = get_files_from_dir(test_dir)
    for test_file in test_files:
        compare_start_time = datetime.now()

        # Step 3: 读取待检测文件特征
        test_features = read_test_features_from_file('%s/%s' % (test_dir, test_file))

        # Step 4: 将待检测特征导入GPU
        d_test_features_in_gpu = copy_test_features_to_gpu(test_features)

        # Step 5: 用GPU计算各个位置的概率
        frames_results = compare_by_gpu(d_test_features_in_gpu, d_all_sources_features_in_gpu, file_features_index_array)

        # Step 6: 找出可能性最高的位置
        calculate_result(test_file, file_index_dict, frames_results, len(test_features), value)

        compare_end_time = datetime.now()
        print('在 %d 个文件中比对 %s 用时: %s\n' % (len(file_index_dict), test_file, (compare_end_time - compare_start_time)))


if __name__ == '__main__':
    source_files_dir = './data/source'  # 比对 所有源的特征值文本 存放路径
    source_files_dir = '/storage/auto_test/source_result'  # 比对 所有源的特征值文本 存放路径
    test_files_dir = './data/test'

    demo_compare(test_files_dir, source_files_dir, 0.05)
