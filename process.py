# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pickle
import gzip
from tqdm import tqdm
import json

meta_data_path_dict = {
    'instruments': "original_data/Recformer_data/meta_Musical_Instruments.json.gz",
    'arts': "original_data/Recformer_data/meta_Arts_Crafts_and_Sewing.json.gz",
    'scientific': "original_data/Recformer_data/meta_Industrial_and_Scientific.json.gz",
    'pet': "original_data/Recformer_data/meta_Pet_Supplies.json.gz",
    'prime_pantry': "original_data/meta_Prime_Pantry.json.gz",
}

def preprocess_interaction(intercation_path, prefix='amazon_books', items=None, cold_ids_file_name=None, output_path=None):
    if prefix == 'amazon_books':
        ratings = pd.read_csv(
            intercation_path,
            sep=",",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
    elif prefix in ['prime_pantry', 'beauty']:
        extract_meta_data(prefix)
        ratings = pd.read_csv(
            intercation_path,
            sep=",",
            names=["item_id", "user_id", "rating", "timestamp"],
        )
    elif prefix in ['arts', 'scientific', 'instruments', 'pet']:
        meta_dict = extract_meta_data(prefix)       
        data = []
        with gzip.open(intercation_path, 'rb') as f:
            for line in f:
                # 解码为字符串并加载成json对象
                record = json.loads(line.decode('utf-8'))
                # 提取需要的字段
                item_id = record['asin']
                if item_id in meta_dict:
                    data.append({
                        "item_id": item_id,
                        "user_id": record["reviewerID"],
                        "timestamp": record["unixReviewTime"]
                    })

        # 将列表转换为 DataFrame，并指定列的顺序
        ratings = pd.DataFrame(data, columns=["item_id", "user_id", "timestamp"])
    
    elif prefix == 'movielens':
        ratings = pd.read_csv(
                intercation_path,
                sep="::",
                names=["user_id", "item_id", "rating", "timestamp"],
                encoding="iso-8859-1",
            )
    
    print(f"{prefix} #data points before filter: {ratings.shape[0]}")
    print(
        # f"{prefix} #user before filter: {len(set(ratings['user_id'].values))}"
        f"{prefix} #user before filter: {len(ratings['user_id'].unique())}"
    )
    print(
        # f"{prefix} #item before filter: {len(set(ratings['item_id'].values))}"
        f"{prefix} #item before filter: {len(ratings['item_id'].unique())}"
    )

    # filter users and items with presence < 5
    item_id_count = (
        ratings["item_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="item_count")
    )
    user_id_count = (
        ratings["user_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="user_count")
    )
    print("Average item count:", round(item_id_count["item_count"].mean(),2))
    print("Average user count:", round(user_id_count["user_count"].mean(),2))
    
    ratings = ratings.join(item_id_count.set_index("unique_values"), on="item_id")
    ratings = ratings.join(user_id_count.set_index("unique_values"), on="user_id")
    if prefix in ['amazon_books', 'arts', 'scientific', 'instruments', 'pet']:                                                                                                        
        ratings = ratings[ratings["item_count"] >= 5]
        ratings = ratings[ratings["user_count"] >= 5]
        ratings = ratings.groupby('user_id').filter(lambda x: len(x['item_id']) >= 5)
    else:
        ratings = ratings[ratings["user_count"] > 3]
    print(f"{prefix} #data points after filter: {ratings.shape[0]}")

    print(
        # f"{prefix} #user after filter: {len(set(ratings['user_id'].values))}"
        f"{prefix} #user after filter: {len(ratings['user_id'].unique())}"
    )
    all_unique_item_ids = set(ratings['item_id'].unique())
    print(
        # f"{prefix} #item after filter: {len(set(ratings['item_id'].values))}"
        f"{prefix} #item after filter: {len(all_unique_item_ids)}"
    )
    user_id_count = (
        ratings["user_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="user_count")
    )
    print("Average user count after filter:", round(user_id_count["user_count"].mean(),2))
    ratings = ratings[['item_id', 'user_id', 'timestamp']]
    ratings.to_csv(f"dataset/{prefix}.csv", index=False, header=True)
    
    ratings_sorted = ratings.sort_values(by=['user_id', 'timestamp', 'item_id'])
    ratings_sorted['user_cumcount'] = ratings_sorted.groupby('user_id').cumcount()
    ratings_sorted['item_cumcount'] = ratings_sorted.groupby('item_id').cumcount()
    ratings_sorted['type'] = 'train'  # 默认设置为 train
    ratings_sorted.loc[ratings_sorted.groupby('user_id')['user_cumcount'].transform('max') - 1 == ratings_sorted['user_cumcount'], 'type'] = 'valid'
    ratings_sorted.loc[ratings_sorted.groupby('user_id')['user_cumcount'].transform('max') == ratings_sorted['user_cumcount'], 'type'] = 'test'
    ratings_sorted = ratings_sorted.drop(columns=['user_cumcount'])
    item_counts = ratings_sorted[ratings_sorted['type'] == 'train']['item_id'].value_counts()

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import numpy as np
    # plt.figure(figsize=(10, 6))
    # bins = np.arange(0, 101, 5) 
    # sns.histplot(item_counts, bins=bins, kde=True)  # kde=True 添加密度曲线
    # plt.xlim(0, 100)  # 设置横轴范围为 0 到 100
    # plt.xlabel("Number of Ratings per Item")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Item Ratings Count")
    # plt.savefig("fig/Frequency.png")

    threshold = 5
    warm_items = set(item_counts[item_counts >= threshold].index)
    cold_items = all_unique_item_ids - warm_items
    strictly_cold_items = cold_items - set(item_counts[item_counts < threshold].index)

    print("len(strictly_cold_items) =", len(strictly_cold_items), ",len(cold_items) =", len(cold_items), ",len(warm_items) =", len(warm_items))
    with open(f"dataset/{prefix}_coldrec_ids.pkl", "wb") as f:
        pickle.dump({"cold_items": cold_items, "strictly_cold_items": strictly_cold_items}, f)

    ratings_sorted['cold_item'] = ratings_sorted['item_id'].apply(
        lambda x: 1 if x in cold_items else 0
    )
    print(len(ratings_sorted))
    ratings_sorted[['item_id', 'user_id', 'timestamp','cold_item']].to_csv(f"dataset/{prefix}.csv", index=False, header=True)

    # 往每个用户序列后加一个交互
    # max_timestamp_plus_1 = ratings['timestamp'].max() + 1
    # # **1. 创建新的 DataFrame**
    # user_ids = ratings['user_id'].unique()
    # new_rows = pd.DataFrame({'user_id': user_ids, 'item_id': 0, 'timestamp': max_timestamp_plus_1})
    # # **2. 追加到原数据集**
    # ratings = pd.concat([ratings, new_rows], ignore_index=True)
    # print(f"{prefix} #data points after adding item_id=0: {ratings.shape[0]}")

    if prefix == 'movielens':
        # 用户冷启动数据集
        ratings = preprocess_user("../generative-recommenders/tmp/ml-1m/users.dat", ratings, items, prefix=prefix)
        print(ratings)
        ratings.to_csv(f"dataset/{prefix}.csv", index=False, header=True)
        # 10%用户截断为7并分入测试集
        num_unique_users = len(set(ratings["user_id"].values))
        user_id_split = int(num_unique_users * 0.9)
        print("user_id_split", user_id_split)
        ratings_data_train = ratings[ratings["user_id"] <= user_id_split]
        ratings_data_train.to_csv(f"dataset/{prefix}-train.csv", index=False, header=True)
        ratings_data_test = ratings[ratings["user_id"] > user_id_split]
        print(f"train num user: {len(set(ratings_data_train['user_id'].values))}")
        print(f"test num user: {len(set(ratings_data_test['user_id'].values))}")

        ratings_sorted = ratings_data_test.sort_values(by=['user_id', 'timestamp', 'item_id'])
        ratings_sorted['row_number'] = ratings_sorted.groupby('user_id').cumcount()
        ratings_sorted = ratings_sorted[ratings_sorted['row_number'] < 10]
        print(ratings_sorted[:21])
        ratings_sorted[['item_id', 'user_id', 'timestamp']].to_csv(f"dataset/{prefix}-test.csv", index=False, header=True)
    

def preprocess_user(user_path, ratings=None, items=None, prefix='movielens'):
    max_id = ratings['item_id'].max()
    min_timestamp = ratings['timestamp'].min()
    print("min_timestamp", min_timestamp)
    print("max_id", max_id)
    if prefix == 'movielens':
        users = pd.read_csv(
                user_path,
                sep="::",
                names=["user_id", "sex", "age_group", "occupation", "zip_code"],
                encoding="iso-8859-1",
            )
        occupation_map = {
            0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student",
            5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer", 9: "homemaker",
            10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
            15: "scientist", 16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
        }
        age_map = {1:  "Under 18",
            18:  "18-24",
            25:  "25-34",
            35:  "35-44",
            45:  "45-49",
            50:  "50-55",
            56:  "56+"}
        sex_map = {
            "M": "Male", "F": "Female"
        }
        users.sex = pd.Categorical(users.sex)
        new_sex_map = {idx: sex_label for idx, sex_label in enumerate(users.sex.cat.categories.tolist())}
        users["sex"] = users.sex.cat.codes

        users.age_group = pd.Categorical(users.age_group)
        new_age_map = {idx: age_label for idx, age_label in enumerate(users.age_group.cat.categories.tolist())}
        users["age_group"] = users.age_group.cat.codes

        users.occupation = pd.Categorical(users.occupation)
        new_occ_map = {idx: occ_label for idx, occ_label in enumerate(users.occupation.cat.categories.tolist())}
        users["occupation"] = users.occupation.cat.codes

        final_age_map = {
            new_idx: age_map[age_code] 
            for new_idx, age_code in new_age_map.items()
        }

        final_sex_map = {
            new_idx: sex_map[sex_code] 
            for new_idx, sex_code in new_sex_map.items()
        }


        users = users[["user_id", 'sex','age_group','occupation']]
        
        dicts = [final_sex_map, final_age_map, occupation_map]  # 绑定外部变量
        for i, col in enumerate(['sex', 'age_group', 'occupation']):
            users[col] += int(max_id + 1)
            
            # ✅ 直接修改原始字典
            dicts[i] = {k + int(max_id + 1): v for k, v in dicts[i].items()}
            
            max_id = users[col].max()
            print(dicts[i])  # 这里打印的就是修改后的字典
            print(max_id)

        users = users.set_index("user_id")
        print(users)


        stacked = users.stack().reset_index()
        stacked.columns = ["user_id", "column", "item_id"]

        result = stacked[["user_id", "item_id"]]
        result['timestamp'] = [min_timestamp - 3, min_timestamp - 2, min_timestamp - 1] * len(users)
        ratings = pd.concat([ratings, result])

        for map_, title in zip(dicts,['sex','age','occupation']):
            map_ = pd.DataFrame.from_dict(map_, orient='index', columns=['description']).reset_index().rename(columns={'index': 'item_id'})
            map_['title'] = title
            items = pd.concat([items, map_])
            print(items)
        items.to_csv("information/ml-1m-user-info.csv", index=False)

        return ratings


def preprocess_item(item_path, prefix='amazon_books', output_path=None):
    if prefix in ['amazon_books', "prime_pantry", 'beauty']:
        data = []
        for line in open(item_path):
            json_data = eval(line)
            # print(json_data.keys())
            item_id = json_data.get('asin', '')
            description = json_data.get('description', '')
            title = json_data.get('title', '')

            data.append({
                'item_id': item_id,
                'description': description,
                'title': title
            })
        df = pd.DataFrame(data)
    elif prefix == 'movielens':
        df = pd.read_csv(
                item_path,
                sep="::",
                names=["item_id", "title", "description"],
                encoding="iso-8859-1",
            )

    df.to_csv(f"information/{prefix}.csv", index=False)
    return df

def extract_meta_data(prefix):
    path = meta_data_path_dict[prefix]
    meta_data = dict()
    data = []
    with gzip.open(path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            attr_dict = dict()
            asin = line['asin']
            # category = ' '.join(line['category'])
            # brand = line['brand']
            title = line['title']

            # attr_dict['title'] = title
            # attr_dict['brand'] = brand
            # attr_dict['category'] = category
            meta_data[asin] = attr_dict
            data.append({
                    'item_id': asin,
                    'description': line.get('description', ''),
                    'title': title
                })
    df = pd.DataFrame(data)
    df.to_csv(f"information/{prefix}.csv", index=False)

    return meta_data
    
if __name__ == '__main__':
    # preprocess_interaction("original_data/ratings_Books.csv", "amazon_books")
    # preprocess_item("original_data/meta_Books.json", "amazon_books")

    # items = preprocess_item("../generative-recommenders/tmp/ml-1m/movies.dat", "information/ml-1m-user-info.csv", "movielens")
    # preprocess_interaction("../generative-recommenders/tmp/ml-1m/ratings.dat", output_path="ml-1m-user-info", prefix="movielens", items=items)

    # preprocess_item("original_data/meta_Prime_Pantry.json", "prime_pantry")
    preprocess_interaction("original_data/Prime_Pantry.csv", "prime_pantry")

    # preprocess_item("original_data/meta_All_Beauty.json", "beauty")
    # preprocess_interaction("original_data/All_Beauty.csv", "beauty")

    preprocess_interaction("original_data/Recformer_data/Pet_Supplies_5.json.gz", "pet")
    preprocess_interaction("original_data/Recformer_data/Musical_Instruments_5.json.gz", "instruments")
    preprocess_interaction("original_data/Recformer_data/Arts_Crafts_and_Sewing_5.json.gz", "arts")
    preprocess_interaction("original_data/Recformer_data/Industrial_and_Scientific_5.json.gz", "scientific")
    

