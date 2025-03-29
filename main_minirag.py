# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import os

from MiniRAG_mydata.minirag import MiniRAG, QueryParam
from MiniRAG_mydata.minirag.llm.hf import (
    hf_model_complete,
    hf_embed,
)
from MiniRAG_mydata.minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse
from tools.qwen_inference import QwenInference


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--workingdir", type=str, default="/pathxxx/MiniRAG_lite_Qwen/MiniRAG_mydata/Old-shige-1")
    parser.add_argument("--datapath", type=str, default="/pathxxx/MiniRAG-mydata/dataset/old_shige")
    args = parser.parse_args()
    return args

args = get_args()
async def qwen_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
):
    # 构建输入消息
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 这里需要根据 Qwen 模型的输入要求来构建输入
    # 假设 Qwen 模型的输入是一个字符串，你可以将消息列表转换为字符串
    input_prompt = ""
    for msg in messages:
        input_prompt += f"<{msg['role']}>{msg['content']}</{msg['role']}>\n"

    # 调用 Qwen 模型进行推理
    result = qwen_inference.generate(
        user_input=input_prompt,
        context=[],
        max_gen_len=kwargs.get("max_new_tokens", 512)  # 假设支持设置最大生成长度
    )

    return result

if args.model == "qwen":
    LLM_MODEL = "/pathxxx/my_weight/qwen2"
else:
    print("Invalid model name")
    exit(1)

# 初始化wo 的加速推理模型
qwen_inference = QwenInference(
    model_path=LLM_MODEL,
    max_seq_len=8192,
    temperature=0.6
)


WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=qwen_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)


# Now indexing
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


WEEK_LIST = find_txt_files(DATA_PATH)
for WEEK in WEEK_LIST:
    id = WEEK_LIST.index(WEEK)
    print(f"{id}/{len(WEEK_LIST)}")
    with open(WEEK) as f:
        rag.insert(f.read())

# A toy query
query = '如果让你写一首诗来描述不要因为时光流逝，而担心客人的鬓发变白，因为内心的纯净和美好是不会改变的。你会怎么写?'
answer = (
    rag.query(query, param=QueryParam(mode="mini")).replace("\n", "").replace("\r", "")
)
print(answer)
