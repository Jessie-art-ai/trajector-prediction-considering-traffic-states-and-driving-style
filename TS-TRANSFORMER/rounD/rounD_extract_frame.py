import os
import pandas as pd

# 设置原文件夹路径
input_folder = r'E:\yjy\roundabout\results\rounD\data_for_pre'

# 设置输出文件夹路径
output_folder = os.path.join(r'E:\yjy\roundabout\results\rounD\data_extract_frame')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 读取轨迹文件数据
        filepath = os.path.join(input_folder, filename)
        data = pd.read_csv(filepath)

        # 根据 frame 列对数据进行抽帧
        new_data = data.iloc[::5, :]

        # 保存处理后的数据
        new_filepath = os.path.join(output_folder, filename)
        new_data.to_csv(new_filepath, index=False)

print('处理完成!')
