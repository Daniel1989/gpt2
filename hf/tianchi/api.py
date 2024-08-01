from gradio_client import Client
import requests
# client = Client("http://127.0.0.1:7860")
# result = client.predict("一个有钱的单身汉必定想要娶妻")

#
result = requests.post("http://127.0.0.1:7860/api/summary", json={"data": ["一个有钱的单身汉必定想要娶妻"]})
print(result.json())