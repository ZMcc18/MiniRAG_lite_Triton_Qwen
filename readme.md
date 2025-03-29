# MiniRAG_lite_Triton_Qwen
## 简介
MiniRAG_lite_Triton_Qwen 是一个创新性的项目，它将轻量级检索增强生成（RAG）技术与集成了 Triton 高效推理框架的 Qwen 大语言模型深度融合。此项目借助高效的检索机制，能够从本地数据集中精准提取相关信息，再依托 Qwen 模型的强大能力进行推理，进而为用户提供更精准、更具针对性的回答。

## 功能特点
- 轻量级 RAG 实现 ：采用轻量级的检索增强生成技术，减少资源消耗，提高检索效率。
- Qwen 模型支持 ：集成 Triton 的高效推理框架的 Qwen 大语言模型，利用其强大的语言生成能力。
- 本地数据集支持 ：可以从本地数据集中检索相关信息，为模型生成提供支持。
- 可配置性 ：通过命令行参数可以配置模型、工作目录和数据集路径等。
- TODO：关于冗余性、逻辑一致性、术语精准性的评估
## 项目结构
```plaintext
MiniRAG_lite_Qwen/
├── Human_machine_evaluation/
│   ├── H_m_consistency_calibration.py
│   ├── Semantic_Consistency.py
│   └── term_accuracy.py
├── MiniRAG_mydata/
│   ├── .gitignore
│   ├── .pre-commit-config.yaml
│   ├── Dockerfile
│   ├── LICENSE
│   ├── MANIFEST.in
│   ├── Old-shige-1/
│   ├── README.md
│   ├── README_CN.md
│   ├── README_JA.md
│   ├── __init__.py
│   ├── assets/
│   ├── dataset/
│   ├── docker-compose.yml
│   ├── graph-visuals/
│   ├── main.py
│   ├── minirag/
│   ├── pyproject.toml
│   ├── reproduce/
│   ├── requirements.txt
│   └── setup.py
├── lite_qwen/
│   └── ...
├── main_minirag.py
├── main_norag.py
├── main_rag.py
└── tools/
    ├── __pycache__/
    ├── apply_weight_convert.py
    └── qwen_inference.py
 ```

## 安装依赖
首先，确保你已经安装了 conda 环境管理工具。然后，创建并激活一个新的虚拟环境：

```bash
conda create -n MiniRAG_lite_Qwen python=3.11
conda activate MiniRAG_lite_Qwen
 ```
```

接着，安装项目所需的依赖：

```bash
pip install -r MiniRAG_mydata/requirements.txt
pip install -r lite_qwen/requirements.txt
 ```

```

## 配置参数
可以通过命令行参数来配置项目的运行参数，例如：

```bash
python main_minirag.py --model qwen --workingdir /path/to/working/dir --datapath /path/to/dataset
 ```
```

- --model ：指定使用的模型，目前支持 qwen2.5-3b-Instruct 。
- --workingdir ：指定工作目录，用于存储中间结果。
- --datapath ：指定数据集路径，用于检索相关信息。
## 运行项目
在完成配置后，可以运行以下命令来启动项目：

```bash
python main_minirag.py
 ```

项目会自动加载数据集，进行索引，并执行一个示例查询。

## 代码说明
### main_minirag.py
项目的主入口文件，负责加载配置、初始化 RAG 实例、进行索引和查询操作。

### qwen_complete 函数
该函数用于调用 Qwen 模型进行高效推理，根据输入的提示信息生成相应的回答。

### MiniRAG 类
实现了轻量级的检索增强生成功能，包括索引构建、查询处理等。

## 注意事项
- 确保 Qwen 模型的路径正确，并且模型文件可以正常加载。
- 数据集路径应包含有效的文本文件，项目会自动递归查找所有 .txt 文件。

# 引用
MiniRAG项目地址: https://github.com/HKUDS/MiniRAG
