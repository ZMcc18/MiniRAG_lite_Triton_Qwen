from tools.qwen_inference import QwenInference

# 初始化接口
model = QwenInference(
    model_path="/home/xd/auditory_eeg/lzxu/triton_project/lite_llama/my_weight/qwen2",
    max_seq_len=2048,
    temperature=0.6
)

context_docs = ["机器学习需要数学基础...", "推荐《深度学习入门》..."]

while True:
    # 获取用户输入
    user_input = input("请输入你的问题（输入 'exit' 退出）：")
    if user_input.lower() == 'exit':
        break

    # 生成回答
    response = model.generate(
        user_input=user_input,
        context=context_docs,
        max_gen_len=512
    )
    print(response)