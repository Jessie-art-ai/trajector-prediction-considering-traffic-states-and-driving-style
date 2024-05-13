import os
import pandas as pd

# 设置要处理的文件夹路径
base_dir = 'E:/yjy/code/Improving-Multi-agent-Trajectory-Prediction-using-Traffic-States-on-Interactive-Driving-Scenarios-master/TS-TRANSFORMER/rounD/process_5fps_2'
sub_dirs = ['train', 'test', 'val']
output_dir = 'process_5fps_2'

# 创建输出文件夹及其子文件夹
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(output_dir, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.mkdir(sub_dir_path)

# 遍历每个子文件夹
for sub_dir in sub_dirs:
    folder_path = os.path.join(base_dir, sub_dir)
    output_sub_dir = os.path.join(output_dir, sub_dir)

    # 遍历文件夹下的所有txt文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(folder_path, filename)
            output_file_path = os.path.join(output_sub_dir, filename)

            # 读取并处理txt文件
            with open(input_file_path, 'r') as f:
                lines = f.readlines()
                processed_lines = []
                for line in lines:
                    processed_line = line.strip().replace('nan', '-1')
                    processed_lines.append(processed_line)

            # 将处理后的内容写入输出文件
            with open(output_file_path, 'w') as f:
                f.write('\n'.join(processed_lines))

print('所有文件处理完成,存储在process_5fps_2文件夹下!')