import streamlit as st
import torch
import time
from transformers import BertTokenizer
from models.bert import BertClassifier
from models.lstm import LstmClassifier

bert_model = torch.load("checkpoints/bert_model.pth")
lstm_model = torch.load("checkpoints/lstm_model.pth")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def bert_predict(tokens):
    start_time = time.time()
    outputs = bert_model(**tokens)
    _, prediction = torch.max(outputs[0], dim=1)
    return prediction, time.time() - start_time

def lstm_predict(tokens):
    start_time = time.time()
    outputs = lstm_model(**tokens)
    _, prediction = torch.max(outputs[0], dim=1)
    return prediction, time.time() - start_time



def write():
    # 设置页面标题
    st.title("情感分析应用")

    # 创建一个输入框，让用户输入文本z
    input_text = st.text_area("输入文本")

    tokens = tokenizer.encode_plus(
        text=input_text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # 创建一个下拉菜单，让用户选择情感分析器
    options = ["Bert", "LSTM"]
    choice = st.selectbox("选择情感分析器", options)

    # 在用户点击“分析”按钮后进行情感分析，并显示结果
    if st.button("分析"):
        if choice == "Bert":
            result, times = bert_predict(tokens)
        elif choice == "LSTM":
            result, times = lstm_predict(tokens)

    if result == 1:
        st.write("积极 (Take {} seconds)".format(round(times, 3)))
    elif result == 0:
        st.write("消极 (Take {} seconds)".format(round(times, 3)))

if __name__ == '__main__':
    write()

