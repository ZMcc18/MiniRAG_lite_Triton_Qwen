# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import os
from minirag import MiniRAG, QueryParam
from minirag.llm.hf import (
    hf_model_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="qwen")
    # parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./Old-shige-1")
    parser.add_argument("--datapath", type=str, default="/home/xd/auditory_eeg/lzxu/triton_project/MiniRAG-mydata/dataset/old_shige")
    # parser.add_argument(
    #     "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    # )
    args = parser.parse_args()
    return args


args = get_args()

if args.model == "qwen":
    LLM_MODEL = "/home/xd/auditory_eeg/lzxu/triton_project/lite_llama/models/qwen2"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
# QUERY_PATH = args.querypath
# OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
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
query = '如果让你写一首诗来描述不要因为时光流逝，而担心客人的鬓发变白，因为内心的纯净和美好是不会改变的。，你会怎么写?'
answer = (
    rag.query(query, param=QueryParam(mode="mini")).replace("\n", "").replace("\r", "")
)
print(answer)
