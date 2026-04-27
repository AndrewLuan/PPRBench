import json
import re
from typing import List, Dict


FORMULA_PATTERN = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)


def find_missing_numeric_ids(data: List[Dict]) -> Dict:
    """
    找出缺失的数字ID（检测ID序列是否连续）
    """
    valid_ids = []
    for item in data:
        if 'question_id' in item and isinstance(item['question_id'], (int, float)):
            valid_ids.append(int(item['question_id']))

    if not valid_ids:
        return {"valid_ids": [], "missing_ids": [], "range": None, "missing_count": 0}

    valid_ids = sorted(set(valid_ids))
    min_id = min(valid_ids)
    max_id = max(valid_ids)
    missing_ids = [i for i in range(min_id, max_id + 1) if i not in valid_ids]

    return {
        "valid_ids": valid_ids,
        "missing_ids": missing_ids,
        "range": {"min": min_id, "max": max_id},
        "missing_count": len(missing_ids)
    }


def extract_formulas(text: str) -> List[str]:
    """
    从文本中提取所有 $$ ... $$ 公式
    """
    return FORMULA_PATTERN.findall(text)


def format_formulas(formulas: List[str]) -> str:
    """
    将公式整理为"逐问评分"标准格式
    """
    if not formulas:
        return ""

    lines = ["【逐问评分】\n"]
    for idx, formula in enumerate(formulas, start=1):
        lines.append(f"公式{idx}，$$ {formula.strip()} $$，1分\n")

    return "\n".join(lines)


def add_formula_scoring(input_file: str, output_file: str, text_key: str = "answer"):
    """
    为 JSON 数据集中的每个条目提取公式并添加 formula_scoring 字段。

    Args:
        input_file: 输入 JSON 文件路径
        output_file: 输出 JSON 文件路径
        text_key: 包含答案文本的字段名（默认 "answer"）
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        text = item.get(text_key, "")
        formulas = extract_formulas(text)
        item["formula_scoring"] = format_formulas(formulas)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"处理完成！结果已保存至: {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python dataset_utils.py <input.json> <output.json> [text_key]")
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]
    text_key = sys.argv[3] if len(sys.argv) > 3 else "answer"

    add_formula_scoring(input_json, output_json, text_key)
