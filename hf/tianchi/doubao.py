from volcenginesdkarkruntime import Ark
from dotenv import load_dotenv
import json
import argparse

load_dotenv()


def prompt(text):
    client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3")
    response = client.chat.completions.create(
        model="ep-20240615044418-lmzkv",
        messages=[
            {"role": "system", "content": 'You are a help assistant.'},
            {"role": "user", "content": text},
        ],
    )
    result = response.choices[0].message.content
    return result


def extend_background():
    prompt_text = '''
    # 角色
    你是一个资深的中文小说专家，能够准确深入地剖析各类中文小说，通过所提供的小说内容，清晰梳理出故事发生的背景、鲜明的人物形象设定以及关键的主要事件，并以规范的 json 格式进行返回。

    ## 技能
    ### 技能 1: 分析小说
    1. 当接收到小说内容后，仔细阅读并理解。
    2. 准确提炼出故事发生的时间、地点、社会环境等背景信息。
    3. 全面且精准地概括出主要人物的性格、外貌、身份等形象设定。
    4. 清晰梳理出小说中的核心主要事件，包括起因、经过和结果。

    ## 限制
    - 只针对提供的小说内容进行分析，不参考其他无关资料。
    - 严格按照 json 格式返回分析结果，如：{"background": "具体背景描述", "actor": "具体人物形象描述", "events": "具体主要事件描述"}
    - 确保返回的内容准确、完整、清晰，不遗漏重要信息。
    '''

    result_list = []
    with open(f"./data/laoshe.json", "r", encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            temp = prompt_text
            temp += '## 小说标题\n' + item["title"] + '\n'
            temp += '## 小说作者\n' + item["author"] + '\n'
            temp += '## 小说内容\n' + item["content"]

            result = prompt(temp)
            print(result)
            result_list.append({
                **json.loads(result),
                **item
            })

    with open(f"./data/laoshe_result.json", "w", encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False)


def generate_summary():
    prompt_text = '''
        # 角色
        你是老舍短篇作品研究专家，能够根据给定的老舍作品名，为用户清晰讲述作品内容。总结包含故事大致背景，且不超过 100 字。
        
        ## 技能
        1. 当用户给出老舍短篇作品名时，先使用工具搜索相关作品内容。
        2. 依据搜索结果，准确概括作品内容，包括故事背景。
        
        ## 限制
         - 只围绕老舍的短篇作品进行回答，不涉及其他内容。
         - 总结内容必须符合给定格式，不超过 100 字。
         - 严格按照 json 格式返回分析结果，如：{"title": "作品名", "summary": "作品的时代、社会背景，不超过 100 字的故事内容总结"}
         
        ## 待分析待作品名
        '''

    result_list = []
    with open(f"./data/laoshe.json", "r", encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            temp = prompt_text
            temp += '\n' + item["title"]

            result = prompt(temp)
            print(result)
            result_list.append({
                **json.loads(result),
                **item
            })

    with open(f"./data/laoshe_summary.json", "w", encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False)


def merge():
    with open(f"./data/laoshe_result.json", "r", encoding='utf-8') as f:
        context = json.load(f)
    with open(f"./data/laoshe_summary.json", "r", encoding='utf-8') as f:
        summary = json.load(f)
    print(len(context), len(summary))
    result = []
    for i, item in enumerate(summary):
        if context[i]["title"] == item["title"]:
            result.append({
                    "instruction": "你是一个熟读各类小说的专家，请根据提供的故事描述，扩展故事背景，人物形象描述，主要事件描述。",
                    "input": item["summary"],
                    "output": "背景:\n"+context[i]["background"]+"\n人物形象:\n"+context[i]["actor"]+"\n主要事件:\n"+context[i]["events"]
            })
        else:
            print("不一致")
    with open(f"./data/laoshe_merge.json", "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="A simple command-line argument parser.")
    parser.add_argument('--name', type=str, help='Name of the user', default='summary')
    args = parser.parse_args()

    if args.name == 'merge':
        merge()
    elif args.name == 'summary':
        generate_summary()
    else:
        extend_background()


if __name__ == "__main__":
    main()
