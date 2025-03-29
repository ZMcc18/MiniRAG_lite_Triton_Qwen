# dataclass装饰器，自动为类添加特殊方法，如__init__，__repr__，__eq__等，用于简化类的定义，自动生成基于类属性的构造函数
# field 用于自定义dataclass中字段的行为，可以设置默认值、类型检查、元数据等
# fields 用于获取dataclass中所有字段的信息，返回一个包含Field对象的列表，每个Field对象代表一个字段
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

@dataclass
class Qwen2Config:
    # 定义模型的配置类，用于存储模型的各种参数
    max_batch_size: int = 4
    max_seq_len: int = 2048

    # 记录该配置文件支持的模型架构名称列表,如Qwen2ForCausalLM
    architectures: Optional[list] = None
    attention_dropout: float = 0.0

    # 词表中表示开始和结束的特殊token id
    bos_token_id: Optional[int] = 151643
    eos_token_id: Optional[int] = 151645
    hidden_act: str = "silu"

    # 常用默认值，用于初始化模型权重矩阵的相关系数
    initializer_range: float = 0.02

    # 模型隐藏层大小, Qwen2.5-1.5B-Instruct
    # 模型隐藏层大小，Qwen2.5-3B-Insruct
    # Transformer隐藏层的维度大小
    hidden_size: int = 2048
    # Feed Forward层的中间维度（通常是hidden_size的4-8倍）
    intermediate_size: Optional[int] = 11008
    # 最大位置编码长度（上下文窗口大小）
    max_position_embeddings: Optional[int] = 32768

    mlp_bias: bool = False
    model_type: str = "qwen2"
    num_heads: int = 16
    num_layers:Optional[int] = 36
    # GQA的分组头数，也就是kvheads，用于实现GQA注意力机制
    num_kv_heads: Optional[int] = 2

    # 公式中得那个小常数
    rms_norm_eps: float = 1e-6
    # ROPE位置编码得缩放配置，用于扩展上下文长度
    rope_scaling: Optional[Dict[str, Any]] = None
    # 作用：RoPE位置编码的基础频率
    # Qwen2特色：使用1,000,000（Llama通常用10,000），更大的theta值能更好处理长上下文
    rope_theta: float = 1000000.0

    torch_dtype: Optional[str] = "bfloat16"
    transformers_version: str = "4.34.1"
    use_cache: bool = True
    vocab_size: Optional[int] = 151936

    # 控制是否将输入词嵌入层和输出词嵌入层的权重绑定
    tie_word_embeddings: bool = True
    # 滑动窗口注意力机制,当为True时，注意力计算只考虑局部窗口内的token
    use_sliding_window: bool = False
    sliding_window: Optional[int] = 4096
    # 多少decode层用滑窗注意，剩下的用全局注意力
    max_window_layers: int = 21
    device:str = 'cuda'

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        
        # 设置默认属性值，上面定义的所有字段都会被初始化为这些默认值
        for field_name, field_def in self.__dataclass_fields__.items():
            setattr(self,field_name,field_def.default)
        
        # 如果提供了 config_dict，从中更新属性
        if config_dict is not None:
            for key, value in config_dict.items():
                # 处理名称映射
                if key == 'num_attention_heads':
                    self.num_heads = value
                elif key == 'num_hidden_layers':
                    self.num_layers = value
                elif key == 'num_key_value_heads':
                    self.num_kv_heads = value
                elif key == 'max_length':
                    self.max_seq_len = value
                else:
                    setattr(self, key, value)

        # 处理额外的关键字参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                setattr(self, key, value)
        self.head_dim = self.hidden_size // self.num_heads
