#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import sys

def convert_amazon_reviews(input_file, output_file, min_interactions=4):
    """
    将原始 Amazon Reviews 数据 (asin, user, rating, unixTime) 转换为 (userID, itemID) 格式，时间顺序。
    并过滤掉交互数小于 min_interactions 的用户。
    """
    # --- 1) 定义映射表 ---
    user2id = {}
    item2id = {}
    user_index = 1
    item_index = 1

    # 用于存放所有记录: { userID: [(itemID, timestamp), ...], ... }
    user_records = {}

    # --- 2) 读取 CSV，构建 user_records ---
    with open(input_file, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin)
        # 每行期望 [asin, reviewerID, rating, unixTime]
        for row in reader:
            # 如果格式不对或数据缺失，跳过
            if len(row) < 4:
                continue

            asin = row[0].strip()       # 商品ID (原始)
            user = row[1].strip()       # 用户ID (原始)
            # rating = float(row[2])    # 评分 (如需要可使用)
            timestamp = int(row[3])     # Unix 时间戳

            # 映射 user -> userID
            if user not in user2id:
                user2id[user] = user_index
                user_index += 1

            # 映射 asin -> itemID
            if asin not in item2id:
                item2id[asin] = item_index
                item_index += 1

            uid = user2id[user]
            iid = item2id[asin]

            if uid not in user_records:
                user_records[uid] = []
            user_records[uid].append((iid, timestamp))

    # --- 3) 过滤交互数 < min_interactions 的用户 ---
    # 先将要删除的用户ID收集起来，然后统一删除
    to_remove = []
    for uid in user_records:
        if len(user_records[uid]) < min_interactions:
            to_remove.append(uid)
    for uid in to_remove:
        del user_records[uid]

    # --- 4) 按时间排序每个用户的交互 ---
    for uid in user_records:
        user_records[uid].sort(key=lambda x: x[1])  # 按 timestamp 排序

    # --- 5) 写出 (userID, itemID) 到目标文件 ---
    with open(output_file, 'w', encoding='utf-8') as fout:
        for uid, interactions in user_records.items():
            for (iid, t) in interactions:
                fout.write(f"{uid} {iid}\n")

    print("转换完成！")
    print("输出文件:", output_file)
    print("最终用户数 =", len(user_records))
    print("Item 数 =", len(item2id))
    print(f"过滤掉交互数 < {min_interactions} 的用户后，剩余用户数为 {len(user_records)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python convert_amazon_reviews.py <input_csv> <output_txt> [min_interactions=4]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    min_interactions = 4
    if len(sys.argv) >= 4:
        min_interactions = int(sys.argv[3])

    convert_amazon_reviews(input_file, output_file, min_interactions)
