import pandas as pd

# 假设您的 csv 文件名为 'data.csv'
# file_path = 'data/libero_elegant_1013/key_moment_intervention/Full-Task-3/episode_summary_stats.csv'
file_path = 'data/libero_elegant_1013/full_intervention/Full-Task-3/episode_summary_stats.csv'

try:
    # 从文件中读取 CSV 数据
    df = pd.read_csv(file_path)

    # 筛选出 Status 为 'Success' 的行
    success_df = df[df['Status'] == 'Success']

    # 检查 success_df 是否为空，以避免在没有成功轮次时出错
    if not success_df.empty:
        # 计算 Intervention_Count 的平均值
        avg_intervention_count = success_df['Intervention_Count'].mean()

        # 计算 Total_Intervention_Time_s 的平均值
        avg_total_intervention_time = success_df['Total_Intervention_Time_s'].mean()

        # 打印结果
        print(f"文件 '{file_path}' 中成功轮次的数据分析结果：")
        print(f"平均 Intervention_Count: {avg_intervention_count}")
        print(f"平均 Total_Intervention_Time_s: {avg_total_intervention_time}")
    else:
        print(f"在文件 '{file_path}' 中没有找到任何成功的轮次。")

except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。请确保文件名正确，并且文件与脚本在同一目录下。")
except KeyError as e:
    print(f"错误：文件中缺少必要的列: {e}。请检查您的 CSV 文件是否包含 'Status', 'Intervention_Count', 和 'Total_Intervention_Time_s' 列。")