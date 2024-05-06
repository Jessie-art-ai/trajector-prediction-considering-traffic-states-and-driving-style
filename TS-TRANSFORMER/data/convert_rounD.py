import os
import numpy as np
import glob
import sys
import subprocess
import argparse

import sys
import os
import glob
import numpy as np
import argparse

sys.path.append(os.getcwd())  # 将当前工作目录添加到模块搜索路径

# 打印当前工作目录以验证路径是否正确
print("当前工作目录:", os.getcwd())

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="rounD")  # 数据集名称，默认为"rounD"
# 请用你的实际绝对路径替换下面的路径
parser.add_argument('--raw_path',
                    default="E:\\yjy\\code\\Improving-Multi-agent-Trajectory-Prediction-using-Traffic-States-on"
                            "-Interactive-Driving-Scenarios-master\\TS-TRANSFORMER\\rounD\\state_5fps")  #
# 原始数据存放路径
parser.add_argument('--out_path',
                    default="E:\\yjy\\code\\Improving-Multi-agent-Trajectory-Prediction-using-Traffic-States-on"
                            "-Interactive-Driving-Scenarios-master\\TS-TRANSFORMER\\rounD\\process_5fps")  # 输出数据存放路径
args = parser.parse_args()

# 处理每种模式（训练、测试、验证）下的数据
for mode in ['train', 'test', 'val']:
    path_pattern = f'{args.raw_path}/{mode}/*.txt'
    raw_files = sorted(glob.glob(path_pattern))
    print("搜索路径模式:", path_pattern)
    print("找到的文件数:", len(raw_files))

    for raw_file in raw_files:
        print("处理文件:", raw_file)
        raw_data = np.loadtxt(raw_file, delimiter=',', dtype=str)

        # 处理帧ID
        # frame_ids = raw_data[:, 0].astype(np.int64)
        # normalized_frame_ids = (frame_ids - frame_ids.min()) // 10
        # raw_data[:, 0] = normalized_frame_ids.astype(str)

        # 准备新的数据格式，初始值为None或适当的默认值
        new_data = np.full((raw_data.shape[0], 17), np.nan, dtype=object)
        # new_data[:, 0] = normalized_frame_ids.astype(int)  # 帧ID整数化
        new_data[:, 0] = raw_data[:, 0].astype(int)
        new_data[:, 1] = raw_data[:, 1].astype(int)  # 个体ID
        new_data[:, 2] = 'Car'  # 默认类型为Car
        new_data[:, 3] = raw_data[:, 4]  # 驾驶风格
        new_data[:, 13] = raw_data[:, 2].astype(float)  # 假设第13列是x坐标
        new_data[:, 15] = raw_data[:, 3].astype(float)  # 假设第15列是y坐标

        out_file_path = os.path.join(args.out_path, mode, os.path.basename(raw_file))
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        np.savetxt(out_file_path, new_data, fmt='%s', delimiter=',', newline='\n')
        print("输出文件路径:", out_file_path)