import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data2_dir = os.path.join(BASE_DIR, 'data2')
json_path = os.path.join(data2_dir, 'News.json')

# 读取 JSON 文件
print("Reading News.json file...")
with open(json_path, 'r', encoding='utf-8') as f:
    data = []
    for line in f:
        try:
            data.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue

print(f"Total articles: {len(data)}")

# 转换为 DataFrame
df = pd.DataFrame(data)

# 统计类别分布
print("\nCategory distribution:")
category_counts = df['category_merged'].value_counts()
print(category_counts)

# 确保 category_merged 列存在
if 'category_merged' not in df.columns:
    df['category_merged'] = df['category']

# 准备数据 - 合并 headline 和 short_description
df['Title'] = df['headline']
df['Description'] = df['short_description']
df['Class Index'] = df['category_merged']

# 创建类别到索引的映射
categories = sorted(df['Class Index'].unique())
category_to_index = {cat: i+1 for i, cat in enumerate(categories)}
df['Class Index'] = df['Class Index'].map(category_to_index)

# 按类别分层拆分数据
train_dfs = []
val_dfs = []
test_dfs = []

for category in categories:
    category_df = df[df['category_merged'] == category]
    if len(category_df) < 10:  # 确保每个类别至少有 10 个样本
        print(f"Warning: Category {category} has only {len(category_df)} samples")
        continue
    
    # 先拆分为训练集和测试集（80% 训练，20% 测试+验证）
    train, temp = train_test_split(category_df, test_size=0.2, random_state=42)
    # 再将测试+验证集拆分为验证集和测试集（各 10%）
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    train_dfs.append(train)
    val_dfs.append(val)
    test_dfs.append(test)

# 合并所有类别的数据
train_df = pd.concat(train_dfs, ignore_index=True)
val_df = pd.concat(val_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

# 打乱顺序
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nSplit results:")
print(f"Train set: {len(train_df)} articles")
print(f"Validation set: {len(val_df)} articles")
print(f"Test set: {len(test_df)} articles")

# 保存为 CSV 文件
train_path = os.path.join(data2_dir, 'train.csv')
val_path = os.path.join(data2_dir, 'val.csv')
test_path = os.path.join(data2_dir, 'test.csv')

# 只保存需要的列
columns_to_save = ['Class Index', 'Title', 'Description']
train_df[columns_to_save].to_csv(train_path, index=False)
val_df[columns_to_save].to_csv(val_path, index=False)
test_df[columns_to_save].to_csv(test_path, index=False)

print(f"\nFiles saved:")
print(f"Train set: {train_path}")
print(f"Validation set: {val_path}")
print(f"Test set: {test_path}")

# 保存类别映射
category_map_path = os.path.join(data2_dir, 'category_map.json')
with open(category_map_path, 'w', encoding='utf-8') as f:
    json.dump(category_to_index, f, indent=2)

print(f"\nCategory map saved to: {category_map_path}")
print("\nProcess completed successfully!")
