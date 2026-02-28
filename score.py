import pandas as pd
from openai import OpenAI
import time
import sys
import dashscope
from dashscope import Generation


prompt = """
你是一名物理老师，任务是根据【评分表】，批改【学生解答】。不要自行加减分。


###输入格式

你会得到二部分内容：
	1.	【评分表】：包含公式编号（例如tag{1},tag{2},tag{3}),公式（或原始公式）,分数(例如1分，2分，3分，4分）。
	2.	【学生解答】：学生的作答，包括文字和公式。


###评分规则
	1.	逐条对照学生的公式和评分表中的公式进行打分
	•	学生解答若出现跟打分表相同公式或物理意义等价、量纲等价的公式，则给分数。
    •	学生解答中已经得分的公式不要重复得分
    •	评分表中已经被获得分数的公式，不再得分
    •	等价公式的判定标准：允许代数变形、符号调整、换元形式，逻辑要完全等价，物理量纲要等价，物理量纲不等价不给分。

	2.	部分得分机制
	•	若某问需要多个公式才能满分，则学生必须写出这些公式才能获得对应分数。
	•	若只写对部分公式，则只给对应的分值。
	•	不额外奖励没有在评分标准中的推导。
	3.	严格遵循评分标准
	•	不根据解题思路长短、表达复杂程度额外加分。
	•	若学生公式完全错误或缺失，不得分。
	4.	文字说明不代替公式
	•	若学生只用文字叙述而未写公式，则不给分。
	•	若文字叙述能表明其明确写出了与公式等价的物理关系（例如“等势面是旋转椭球”），可给相应分数。


###输出格式

请输出如下结构：
	1.	【逐问评分】：
	•	按照题目顺序，列出每个小问的得分情况，说明该小问哪些公式被识别正确并获得对应分数。
	•	使用 “式①” “式②” 对应评分标准中的标号。
	•	如果未出现，标注“未出现”。
	2.	【总分】：
	•	汇总所有小问得分，总结学生该题的最终得分。用\\boxed{}表示。


###输出示例

（下面是示例，不要照搬分数）

【逐问评分】

第(1)问（5分）：
- 学生写出了 φ(r,z) 的积分表达式，等价于标准答案式① → 得5分。

第(2)问（14分）：
- 学生指出等势面为旋转椭球 → 得4分。
- 未出现式② → 0分。
- 出现了等价于式③的表达式 → 得4分。
- 出现了等价于式④的表达式 → 得4分。
本问合计：12分。

第(3)问（8分）：
- 出现了等价于式⑤ → 得2分。
- 未出现式⑥ → 0分。
- 出现了等价于式⑦ → 得4分。
本问合计：6分。

第(4)问（13分）：
- 出现了等价于式⑧ → 得2分。
- 出现了等价于式⑨ → 得2分。
- 未出现式⑩ → 0分。
- 出现了等价于式⑪ → 得3分。
- 未出现式⑫ → 0分。
本问合计：7分。

【总分】
本题总分：$5 + 12 + 6 + 7 = \\boxed{30}$。
"""

client = OpenAI(
    api_key="<Your_api_key>",
    base_url="<Your_api_provider>",
)


def remove_think_block(text: str) -> str:
    """
    删除 <think>...</think> 及其内部所有内容。
    若存在未闭合的 <think>（即有 <think> 但无 </think>），返回空串。
    """

    open_tag = "<think>"
    close_tag = "</think>"

    start = text.find(open_tag)
    if start == -1:

        return text

    end = text.find(close_tag, start + len(open_tag))
    if end == -1:

        return ""

    return text[:start] + text[end + len(close_tag):]


def remove_repeated_spans(
    text: str,
    min_span_len: int = 10,
    max_span_len: int = 600,
    min_repeats: int = 2
) -> str:
    """
    删除字符串中连续重复的片段（常见于 LLM 幻觉重复）。

    Parameters
    ----------
    text : str
        输入字符串
    min_span_len : int
        最小检测片段长度（字符数），过小容易误删
    max_span_len : int
        最大检测片段长度（限制复杂度）
    min_repeats : int
        至少连续重复多少次才判定为幻觉

    Returns
    -------
    str
        去除重复片段后的字符串
    """

    i = 0
    n = len(text)
    result = []

    while i < n:
        found = False

        for span_len in range(
            min(max_span_len, (n - i) // min_repeats),
            min_span_len - 1,
            -1
        ):
            span = text[i: i + span_len]
            repeats = 1

            while (
                i + repeats * span_len + span_len <= n
                and text[
                    i + repeats * span_len:
                    i + (repeats + 1) * span_len
                ] == span
            ):
                repeats += 1

            if repeats >= min_repeats:
                result.append(span)
                i += repeats * span_len
                found = True
                break

        if not found:
            result.append(text[i])
            i += 1

    return "".join(result)


def call_lm(ref_score: str, ref_studentanswer: str, model: str = "qwen3-235b-a22b-instruct-2507") -> str:

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",
            "content": f"根据下面的评分表重新开始评分，【评分表】：\n{ref_score}\n【学生解答】：\n{remove_repeated_spans(ref_studentanswer)}"}
    ]
    try:
        print("正在调用模型...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8200,
            temperature=0.2
        )
        output_text = response.choices[0].message.content
        print("------output:")
        print(output_text)
        return output_text.strip()
    except Exception as e:
        print(f"Exception during OpenAI API call: {e}")
        return "[EXCEPTION]"


def process_excel(input_file: str, output_file: str, ):
    """
    读取 Excel，对每行的问题调用LLM，保存答案到新列并输出新文件
    """
    answer_column = "answer"
    scorestandard_column = "score-standard"
    # 读取 Excel
    df = pd.read_excel(input_file)

    if answer_column not in df.columns:
        raise ValueError(
            f"列 '{answer_column}' 不存在于 Excel 文件中。可用列：{list(df.columns)}")
    if scorestandard_column not in df.columns:
        raise ValueError(
            f"列 '{scorestandard_column}' 不存在于 Excel 文件中。可用列：{list(df.columns)}")

    answers = []

    for idx, row in df.iterrows():
        answerinexcel = str(row[answer_column]).strip()
        scorestandardinexcel = str(row[scorestandard_column]).strip()

        print(f"Processing row {idx + 1}...")
        answerinexcel = remove_think_block(answerinexcel)
        if answerinexcel == "":
            answers.append("\"\\boxed{0}\"")
            continue
        answerinexcel = remove_repeated_spans(answerinexcel)
        print(len(answerinexcel))
        answer = call_lm(scorestandardinexcel, answerinexcel)

        answers.append("\""+answer+"\"")

        time.sleep(1)
    df["scoreresult"] = answers

    df.to_excel(output_file, index=False)
    print(f"处理完成！结果已保存至: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python script.py <input.xlsx> <output.xlsx>")
        sys.exit(1)

    input_excel = sys.argv[1]
    output_excel = sys.argv[2]
    process_excel(input_excel, output_excel)
