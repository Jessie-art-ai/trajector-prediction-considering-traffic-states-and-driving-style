import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 设置输入文件夹路径
input_folder = r'E:\yjy\roundabout\results\rounD\data_extract_frame'

# 设置输出文件夹路径
output_folder = r'../rounD'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
train_folder = os.path.join(output_folder, 'data_5fps/train')
test_folder = os.path.join(output_folder, 'data_5fps/test')
val_folder = os.path.join(output_folder, 'data_5fps/val')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
if not os.path.exists(val_folder):
    os.makedirs(val_folder)
else:
    pass

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 读取数据
        filepath = os.path.join(input_folder, filename)
        data = pd.read_csv(filepath)

        # 根据 trackId 划分数据集
        unique_track_ids = data['trackId'].unique()
        train_track_ids, temp_track_ids = train_test_split(unique_track_ids, test_size=0.3, random_state=42)
        test_track_ids, val_track_ids = train_test_split(temp_track_ids, test_size=2/3, random_state=42)

        # 根据划分好的 trackId 获取对应的数据
        train_data = data[data['trackId'].isin(train_track_ids)]
        test_data = data[data['trackId'].isin(test_track_ids)]
        val_data = data[data['trackId'].isin(val_track_ids)]

        # 输出划分后的数据集
        train_data.to_csv(os.path.join(train_folder, os.path.splitext(filename)[0] + '_train.csv'), index=False)
        test_data.to_csv(os.path.join(test_folder, os.path.splitext(filename)[0] + '_test.csv'), index=False)
        val_data.to_csv(os.path.join(val_folder, os.path.splitext(filename)[0] + '_val.csv'), index=False)

        print(f"处理完成: {filename}")
        print("训练集大小:", train_data.shape[0])
        print("测试集大小:", test_data.shape[0])
        print("验证集大小:", val_data.shape[0])
        pass

