# lite_qwen/inference.py
import torch
from typing import Optional, List, Dict, Union
from lite_qwen.utils.prompt_templates import get_prompter
from lite_qwen.generate_stream import GenerateStreamText
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

class QwenInference:
    def __init__(
        self,
        model_path: str,
        max_seq_len: int = 2048,
        max_gpu_num_blocks = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_model: bool = True,
        triton_weight: bool = True,
        compiled_model: bool = False
    ):
        """
        初始化推理框架
        Args:
            model_path: 模型权重路径
            max_seq_len: 模型最大上下文长度
            max_gpu_num_blocks: GPU显存块数量限制
            temperature: 生成温度
            top_p: 核采样阈值
            device: 运行设备
        """
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        
        # 初始化提示模板构建器
        short_prompt = max_seq_len <= 1024
        self.prompter = get_prompter("qwen2", model_path, short_prompt)
        
        # 初始化生成器
        self.generator = GenerateStreamText(
            checkpoints_dir=model_path,
            tokenizer_path=model_path,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            load_model=load_model,
            compiled_model=compiled_model,
            triton_weight=triton_weight,
            device=device,
        )

    def build_prompt(self, user_input: str, context: Union[str, List[str]] = None) -> str:
        """
        构建符合Qwen2.5格式的Prompt
        Args:
            user_input: 用户输入的问题
            context: 从RAG检索的上下文（支持字符串或列表）
        Returns:
            格式化后的完整Prompt
        """
        if isinstance(context, list):
            context = "\n".join([f"参考资料{i+1}: {doc}" for i, doc in enumerate(context)])
        
        # 插入系统提示和用户输入
        self.prompter.insert_prompt(user_input)
        if context:
            system_msg = f"请根据以下资料回答问题：\n{context}"
            self.prompter.system_prompt = system_msg
        
        return self.prompter.model_input

    def generate(
        self,
        user_input: str,
        context: Union[str, List[str]] = None,
        max_gen_len: Optional[int] = 1024,
        stream: bool = False
    ) -> Union[str, List[Dict]]:
        """
        生成回答
        Args:
            user_input: 用户输入的问题
            context: RAG检索的上下文（可选）
            max_gen_len: 最大生成长度
            stream: 是否流式输出
        Returns:
            若stream=True，返回生成器；否则返回完整文本
        """
        prompt = self.build_prompt(user_input, context)
        prompts = [prompt]
        
        if stream:
            # 返回流式生成器
            return self.generator.text_completion_stream(
                prompts,
                temperature=self.temperature,
                top_p=self.top_p,
                max_gen_len=max_gen_len,
            )
        else:
            # 直接生成完整结果
            outputs = []
            for batch in self.generator.text_completion_stream(prompts, self.temperature, self.top_p, max_gen_len):
                outputs.append(batch[0]['generation'])
            return outputs[-1]  # 返回最终结果