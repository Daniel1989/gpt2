import requests
from bs4 import BeautifulSoup
import json

urls = [
    {
        "prefix": 'http://www.millionbook.com/mj/l/laoshe/gj/',
        "suffix": 'index.html'
    },
    {
        "prefix": 'http://www.millionbook.com/mj/l/laoshe/hcj/',
        "suffix": 'index.html'
    },
    {
        "prefix": 'http://www.millionbook.com/mj/l/laoshe/hzj/',
        "suffix": 'index.html'
    },
    {
        "prefix": 'http://www.millionbook.com/mj/l/laoshe/jw/',
        "suffix": 'index.html'
    },
    {
        "prefix": 'http://www.millionbook.com/mj/l/laoshe/pxj/',
        "suffix": 'index.html'
    },
    {
        "prefix": 'http://www.millionbook.com/mj/l/laoshe/yhj/',
        "suffix": 'index.html'
    },
]

merged = []
for item in urls:
    response = requests.get(item["prefix"] + item["suffix"])
    content = BeautifulSoup(response.text, 'html.parser')
    links = content.find_all('td')
    for link in links:
        a = link.find('a')
        if a is not None and a['href'].startswith('0'):
            url = item["prefix"] + a['href']
            print("url", url)
            article = requests.get(url)
            article.encoding = 'GB2312'
            article_content = BeautifulSoup(article.text, 'html.parser')
            article_content = article_content.find('td', class_='tt2')
            titles = article_content.find_all('center')
            article_info = {}
            if titles[0].text.strip() == "序":
                continue
            article_info["title"] = titles[0].text.strip()
            article_info["author"] = titles[1].text.replace("作者：", "").strip()
            contents = str(article_content).split("<br/>")
            content_list = []
            for content in contents[2:-1]:
                content_list.append(content.strip())
            article_info["content"] = "\n".join(content_list)
            if len(content_list) == 0:
                print("异常url", url)
                continue
            merged.append(article_info)
        else:
            pass
with open(f"./data/laoshe.json", "w", encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=4)