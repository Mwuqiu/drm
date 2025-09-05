import json
import os
import pandas as pd

# 构建文件名列表
files = [f"task_{i}_episode_annotations.json" for i in range(10)]

# 用于存储结果的列表
results = []

# 逐一处理文件
for filename in files:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)

            # 计算 true 和 false 的数量
            true_count = sum(1 for value in data.values() if value is True)
            false_count = sum(1 for value in data.values() if value is False)
            total_count = len(data)

            # 计算比例
            if false_count > 0:
                ratio = true_count / false_count
            else:
                ratio = float('inf')  # 如果没有坏数据，比例为无穷大

            results.append({
                "任务": filename,
                "数据总量": total_count,
                "好数据 (true)": true_count,
                "坏数据 (false)": false_count,
                "好/坏数据比例": ratio
            })
    except FileNotFoundError:
        print(f"警告：文件未找到 - {filename}")
    except json.JSONDecodeError:
        print(f"警告：无法解析文件 - {filename}")

# 如果成功处理了任何文件，则创建并显示表格
if results:
    df = pd.DataFrame(results)

    # 计算总计行
    total_row = {
        "任务": "总计",
        "数据总量": df["数据总量"].sum(),
        "好数据 (true)": df["好数据 (true)"].sum(),
        "坏数据 (false)": df["坏数据 (false)"].sum(),
    }

    if total_row["坏数据 (false)"] > 0:
        total_row["好/坏数据比例"] = total_row["好数据 (true)"] / total_row["坏数据 (false)"]
    else:
        total_row["好/坏数据比例"] = float('inf')

    # 添加总计行
    df.loc[len(df.index)] = total_row

    print("数据分析结果:")
    print(df.to_string())
else:
    print("没有处理任何数据。")