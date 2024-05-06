import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv(r'E:\yjy\roundabout\results\rounD\data_for_pre\8_tra_new.csv')

# 根据 trackId 划分数据集
unique_track_ids = data['trackId'].unique()

# 随机划分 trackId
train_track_ids, temp_track_ids = train_test_split(unique_track_ids, test_size=0.3, random_state=42)
test_track_ids, val_track_ids = train_test_split(temp_track_ids, test_size=2/3, random_state=42)

# 根据划分好的 trackId 获取对应的数据
train_data = data[data['trackId'].isin(train_track_ids)]
test_data = data[data['trackId'].isin(test_track_ids)]
val_data = data[data['trackId'].isin(val_track_ids)]

train_data.to_csv('../rounD/data/train/rounD_8_train.csv')
test_data.to_csv('../rounD/data/test/rounD_8_test.csv')
val_data.to_csv('../rounD/data/val/rounD_8_val.csv')

# 输出划分后的数据集大小
print("训练集大小:", train_data.shape[0])
print("测试集大小:", test_data.shape[0])
print("验证集大小:", val_data.shape[0])
