import gradio as gr
import torch
from transformers import pipeline

model="/Users/caoxiaopeng/Desktop/gpt2/hf/tianchi/merged_model"
gen = pipeline("text-generation", model, device=torch.device('mps'), max_new_tokens=800)


def summary(text):
    result = gen(text)
    return result[0]["generated_text"]

with gr.Blocks() as demo:
    input_text = gr.Textbox(placeholder="输入", lines=4)
    output_text = gr.Textbox(label="输出")
    btn = gr.Button("生成")
    btn.click(summary, input_text, output_text)

demo.launch()