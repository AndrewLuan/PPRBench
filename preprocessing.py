import pandas as pd


def read_csv_stats(filepath):
    """
    读取CSV文件并显示行数、列数和列标签，同时处理常见的CSV问题

    Args:
        filepath (str): CSV文件路径
    """
    try:
        df = pd.read_csv(filepath,
                         sep=None,
                         engine='python',
                         skipinitialspace=True)

        df = df.dropna(axis=1, how='all')

        rows = len(df)
        cols = len(df.columns)

        print(f"文件 '{filepath}' 的统计信息：")
        print(f"行数: {rows}")
        print(f"列数: {cols}")
        print("\n列标签:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")

        print("\n数据预览（前5行）:")
        print(df.head())

    except Exception as e:
        print(f"读取文件时出错: {e}")


def remove_empty_id_rows(input_file, output_file):
    """
    删除CSV文件中id为空的行并保存为新文件

    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
    """
    try:
        df = pd.read_csv(input_file)

        print(f"原始文件行数: {len(df)}")

        df_cleaned = df.dropna(subset=['id'])

        print(f"清理后文件行数: {len(df_cleaned)}")
        print(f"删除的行数: {len(df) - len(df_cleaned)}")

        df_cleaned.to_csv(output_file, index=False)
        print(f"已保存到: {output_file}")

    except Exception as e:
        print(f"处理文件时出错: {e}")


remove_empty_id_rows(
    "<Path_to_input_csv>", "<Path_to_output_csv>")
