import wandb
import sys
import numpy as np
import pandas as pd

# Example RUN_PATHs (replace with your actual W&B run paths)
# RUN_PATH = "your-wandb-team/CRPO/run-id-1"
# RUN_PATH = "your-wandb-team/CRPO/run-id-2"
# RUN_PATH = "your-wandb-team/CRPO/run-id-3"
# RUN_PATH = "your-wandb-team/CRPO/run-id-4"
# RUN_PATH = "your-wandb-team/CRPO/run-id-5"
# RUN_PATH = "your-wandb-team/CRPO/run-id-6"
# RUN_PATH = "your-wandb-team/CRPO/run-id-7"
# RUN_PATH = "your-wandb-team/CRPO/run-id-8"
RUN_PATH = "your-wandb-team/CRPO/your-run-id"


ACCURACY_KEYS = [
    "val-core/OlympiadBench/reward/mean@1",
    "val-core/Minerva/reward/mean@1",
    "val-core/MATH/reward/mean@1",
    "val-core/AMC/reward/mean@1",
    "val-core/AIME24/reward/mean@1"
]

api = wandb.Api()

try:
    print(f"⏳ 正在连接并获取实验 '{RUN_PATH}' 的数据...")
    run = api.run(RUN_PATH)
    print(f"✅ 成功连接到实验: {run.name}")
except Exception as e:
    print(f"❌ 错误：无法获取实验 '{RUN_PATH}'。")
    print(f"   请检查路径是否正确，以及你是否有权限访问。")
    print(f"   W&B API 报错: {e}")
    sys.exit(1)

best_step = -1
max_avg_accuracy = -1.0
best_accuracies = {}

valid_points_list = []

history_keys = ACCURACY_KEYS + ["_step"]
history = run.scan_history(keys=history_keys)

print("🔎 开始扫描实验历史记录")

for row in history:
    if all(key in row and not np.isnan(row[key]) for key in ACCURACY_KEYS):
        valid_points_list.append(row)

print(f"✅ 收集完成！共找到 {len(valid_points_list)} 个符合条件的记录事件。")

df = pd.DataFrame(valid_points_list)

df_unique = df.drop_duplicates(subset=['_step'] + ACCURACY_KEYS)
print(f"✅ 去重完成！剩余 {len(df_unique)} 个唯一的、完整的记录点。")

df_unique['avg_accuracy'] = df_unique[ACCURACY_KEYS].mean(axis=1)
top_10 = df_unique.sort_values(by='avg_accuracy', ascending=False).head(10)

print("\n" + "="*80)
print("🏆 分析完成：以下是Top {} 排名 🏆".format(len(top_10)))

headers = ["Rank", "Step", "Avg Accuracy"] + ACCURACY_KEYS
table_data = []
rank = 1
for index, row in top_10.iterrows():
    data_row = [f"#{rank}", row['_step'], f"{row['avg_accuracy']:.4f}"]
    for key in ACCURACY_KEYS:
        data_row.append(f"{row.get(key, 0):.4f}")
    table_data.append(data_row)
    rank += 1

col_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + table_data))]
header_line = " | ".join(header.ljust(width) for header, width in zip(headers, col_widths))
print("\n" + header_line)
print("-+-".join("-" * width for width in col_widths))
for row in table_data:
    data_line = " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths))
    print(data_line)

print(f"\n🔗  在W&B中查看该实验: {run.url}")
print("="*80)
