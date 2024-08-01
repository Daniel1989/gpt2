import json
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_summary_with_retry(text):
    return text

def process_texts(texts):
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_text = {}
        for text in tqdm(texts, desc="Submitting tasks", total=len(texts)):
            future = executor.submit(get_summary_with_retry, text)
            future_to_text[future] = text
            time.sleep(0.2)  # 控制每0.5秒提交一个任务

        for future in tqdm(as_completed(future_to_text), total=len(texts), desc="Processing tasks"):
            text = future_to_text[future]
            try:
                summary = future.result()
                results.append((text, summary))
            except Exception as e:
                print(f"Failed to process text: {text}. Error: {e}")

    return results


def process_texts(texts):
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_text = {}
        for text in tqdm(texts, desc="Submitting tasks", total=len(texts)):
            future = executor.submit(get_summary_with_retry, text)
            future_to_text[future] = text
            time.sleep(0.2)  # 控制每0.5秒提交一个任务

        for future in tqdm(as_completed(future_to_text), total=len(texts), desc="Processing tasks"):
            text = future_to_text[future]
            try:
                summary = future.result()
                results.append((text, summary))
            except Exception as e:
                print(f"Failed to process text: {text}. Error: {e}")

    return results


def build_dataset(novel, texts):
    # 目标 老舍短文集 http://www.millionbook.com/mj/l/laoshe/index.html
    # 欧亨利
    # 爱伦坡
    instruction_prompt = "你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。"
    dataset = []
    dataset_error = []

    # 使用多线程处理文本
    processed_texts = process_texts(texts)

    for text, summary in processed_texts:
        if summary:
            dataset.append({
                "instruction": instruction_prompt,
                "input": summary,
                "output": text
            })
        else:
            dataset_error.append(text)

    # 保存数据集
    os.makedirs('./data', exist_ok=True)
    with open(f"./data/{novel}.json", "w", encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    with open(f"./data/{novel}_error.txt", "w", encoding='utf-8') as f:
        json.dump(dataset_error, f, ensure_ascii=False, indent=4)

    return dataset