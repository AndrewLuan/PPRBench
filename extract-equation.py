import pandas as pd
import dashscope
from dashscope import Generation
import time
from openai import OpenAI
import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor


client = OpenAI(
    api_key="<Your_api_key>",
    base_url="<Your_api_provider>",
)

prompt = """
你是一名物理老师，任务是根据【标准答案】，生成评分表。


###输入格式

你会得到下面输入：
	1.	【标准答案】：包含公式编号（例如tag{1},tag{2},tag{3}， （1），（2），（3）),公式（或原始公式）,分数(例如1分，2分，3分，4分）。最后是每个公式编号对应的分数



###评分规则
	1.	逐条对标准答案进行分析，找出其中的小问编号，一般格式例如（1),(2),(3),(1.1),(1.2),(1.3),然后找出该小问的所有带编号的公式和对应的分数，公式编号一般格式是tag{数字}，例如tag{1}，tag{2}，tag{3}，公式中间可能有空格不要断开成两个公式，一个公式都以公式编号tag{数字}结尾
	


###输出格式

请输出如下结构：
	1.	【逐问评分】：
	•	按照题目顺序，列出每个小问的公式编号（例如公式1，公式2，公式3），原始公式（保留输入中的原本格式，不要变形， 保留公式末尾的tag{数字}），和分数。




###输出示例

（下面是示例，不要照搬分数）

【逐问评分】

第(1)问（5分）：
公式1，y=a+b, 2分
公式2，y=a+b, 3分

第(2)问（14分）：
公式1，y=a+b, 10分
公式2，y=a+b, 4分

"""


def call_gpt4(answer: str, model: str = "qwen3-235b-a22b") -> str:

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"根据下面的标准答案重新开始，忽视之前所有的标准答案和数据，【标准答案】：\n{answer}\n"}
    ]
    try:
        print("正在调用 Qwen 模型...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8000,
            temperature=0.3
        )
        output_text = response.choices[0].message.content
        print("------output:")
        print(output_text)
        return output_text.strip()
    except Exception as e:
        print(f"Exception during OpenAI API call: {e}")
        return "[EXCEPTION]"


executor = ThreadPoolExecutor(max_workers=8)


async def call_gpt4_async(answer: str, model: str = "qwen3-235b-a22b") -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        call_gpt4,
        answer,
        model
    )


def process_excel(input_file: str, output_file: str, question_column: str = "solution_grading"):

    df = pd.read_excel(input_file)

    if question_column not in df.columns:
        raise ValueError(
            f"列 '{question_column}' 不存在于 Excel 文件中。可用列：{list(df.columns)}")

    answers = []

    for idx, row in df.iterrows():
        questioninexcel = str(row[question_column]).strip()

        if not questioninexcel or questioninexcel.lower() in ["nan", "none", ""]:
            answers.append("")
            continue

        print(f"Processing row {idx + 1}: {questioninexcel[:60]}...")

        answer = call_gpt4(questioninexcel)

        answers.append(answer)

        time.sleep(10)

    df["answers"] = answers

    df.to_excel(output_file, index=False)
    print(f"处理完成！结果已保存至: {output_file}")


def process_json(input_file: str, output_file: str):
    """
    逐条处理 JSON，每生成一次答案就立即写入文件
    保证：
    1. 输入 JSON 的所有键全部保留
    2. 输出 JSON 可读（缩进 + 换行）
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write("[\n")

        for idx, row in enumerate(data):

            answer_json = str(row.get("answer", "")).strip()
            crite_json = str(row.get("scoring_criteria", "")).strip()

            if not answer_json or answer_json.lower() in ["nan", "none", ""]:
                row["formulas"] = ""
            else:
                question_json = answer_json + "\n" + crite_json
                print(f"Processing row {idx + 1}: {question_json[:60]}...")

                try:
                    result = call_gpt4(question_json)
                except Exception as e:
                    print(f"Error at row {idx + 1}: {e}")
                    result = "[ERROR]"

                row["formulas"] = result
                print(result + "\n")

                time.sleep(10)

            json_str = json.dumps(row, ensure_ascii=False, indent=2)

            if idx > 0:
                fout.write(",\n")

            fout.write(json_str)
            fout.flush()

        fout.write("\n]\n")

    print(f"处理完成！结果已实时写入（可读 JSON）: {output_file}")


def get_completed_count(output_file: str) -> int:
    """
    返回 output JSON 中已完成的条目数量
    """
    if not os.path.exists(output_file):
        return 0

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data)
    except Exception:
        return 0


async def process_json_parallel(
    input_file: str,
    output_file: str,
    max_concurrency: int = 4
):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    completed = get_completed_count(output_file)
    total = len(data)

    print(f"检测到已完成 {completed}/{total} 条，自动跳过")

    semaphore = asyncio.Semaphore(max_concurrency)
    lock = asyncio.Lock()

    if completed == 0:
        fout = open(output_file, 'w', encoding='utf-8')
        fout.write("[\n")
    else:
        fout = open(output_file, 'r+', encoding='utf-8')
        fout.seek(0, os.SEEK_END)
        fout.seek(fout.tell() - 2)
        fout.truncate()
        fout.write(",\n")

    fout.flush()

    tasks = []
    for idx in range(completed, total):
        row = data[idx]
        task = asyncio.create_task(
            process_one(idx, row, fout, semaphore, lock)
        )
        tasks.append(task)

    await asyncio.gather(*tasks)

    fout.write("\n]\n")
    fout.close()

    print(f"处理完成：{completed} → {total}")


async def process_one(
    idx: int,
    row: dict,
    fout,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock
):
    async with semaphore:
        answer_json = str(row.get("answer", "")).strip()
        crite_json = str(row.get("scoring_criteria", "")).strip()

        if not answer_json or answer_json.lower() in ["nan", "none", ""]:
            row["formulas"] = ""
        else:
            question_json = answer_json + "\n" + crite_json
            print(f"[Task {idx+1}] calling API...")

            try:
                result = await call_gpt4_async(question_json)
            except Exception as e:
                print(f"[Task {idx+1}] ERROR: {e}")
                result = "[ERROR]"

            row["formulas"] = result

        async with lock:
            json_str = json.dumps(row, ensure_ascii=False, indent=2)
            fout.write(",\n" + json_str)
            fout.flush()

        print(f"[Task {idx+1}] done")

if __name__ == "__main__":
    input_json = "<Path_to_input_dataset>"
    output_json = "<Path_to_output_json>"

    asyncio.run(
        process_json_parallel(
            input_json,
            output_json,
            max_concurrency=5
        )
    )
