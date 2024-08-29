import csv
import os

import pandas as pd
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
# Load the CSV file into a DataFrame
# data = [
#     ['name', 'title', 'lines'],
#     ['123', 'sdf', 10],
#     ['123', 'sdf', 101],
#     ['123', 'sdf', 110],
# ]
# with open("basic.csv", mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data)
#     print("创建csv文件成功")

df4 = pd.read_csv("basic_batch_4.csv")
df16 = pd.read_csv("basic_batch_16.csv")
df32 = pd.read_csv("basic_batch_32.csv")
df32_1cyc = pd.read_csv("basic_batch_32_direct_gpu.csv")
df32_1cyc_126 = pd.read_csv("basic_batch_32_1cycle_multiple_worker.csv")

df64 = pd.read_csv("basic_batch_64.csv")
key = 'step'
print(df32[key].describe())
print(df32[key].sum()/60, df32[key].mean(), df32[key].median())
print(df32_1cyc[key].describe())
print(df32_1cyc[key].sum()/60, df32_1cyc[key].mean(), df32_1cyc[key].median())
print(df32_1cyc_126[key].describe())
print(df32_1cyc_126[key].sum()/60, df32_1cyc_126[key].mean(), df32_1cyc_126[key].median())