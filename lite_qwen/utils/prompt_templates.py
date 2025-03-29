from typing import List

class BasePrompter:
    """用于构建模型的提示词 (Prompt) 模板和管理对话流程"""
    def __init__(
        self,
        system_inst,
        role1,
        role2,
        sen_spliter="\n",
        qa_spliter="\n",
        colon=":",
        decorator: List[str] = None,
    ):  
        self.system_inst = system_inst  # System Instruction
        self.role1 = role1  # The name of USER
        self.role2 = role2  # The name of AI-Assistant
        self.sen_spliter = sen_spliter  # How to split system/user/assistant outputs
        self.qa_spliter = qa_spliter  # How to split Q&A rounds
        self.decorator = decorator
        self.colon = colon

        if self.decorator == None:
            self.starter = ""
            self.stopper = ""
        else:
            self.starter = self.decorator[0]
            self.stopper = self.decorator[1]

        if self.system_inst == None:
            self.template = (
                self.starter
                + self.role1
                + self.colon
                + " {prompt}"
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role2
                + self.colon
            )
        else:
            self.template = (
                self.starter
                + self.system_inst
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role1
                + self.colon
                + " {prompt}"
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role2
                + self.colon
            )
        self.model_input = None

    def insert_prompt(self, prompt: str):
        """插入新的用户输入，并更新模型输入"""
        self.model_input = self.template.format(prompt=prompt)

    def update_template(self, outputs, chunk_prefilling=0):
        # 动态更新对话模板 ，用于处理多轮对话场景
        if chunk_prefilling:
            self.template = (
                self.role1
                + ": {prompt}"
                + self.stopper
                + self.sen_spliter  # blank space
                + self.starter
                + self.role2
                + ":"
            )
        else:
            self.template = (
                self.model_input
                + " "
                + outputs.strip()
                + self.stopper
                + self.qa_spliter
                + self.starter
                + self.role1
                + ": {prompt}"
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role2
                + ":"
            )
        self.model_input = None

class Qwen2Prompter(BasePrompter):
    def __init__(self):
        # 在 Qwen2 的提示格式下，system_inst 将包含系统信息（如角色设定）
        system_inst = "You are lzx, created by Alibaba Cloud. You are a helpful assistant."
        
        # role1 用作 user 信息块的起始标记，这里不需要额外标记，只需在模板中插入即可
        # role2 用作 assistant 起始标记
        # 我们在构造时，会通过 template 来定义最终的格式。
        
        role1 = "<|im_start|>user\n"    # 用户块开始 
        role2 = "<|im_start|>assistant\n"  # 助手块开始
        sen_spliter = "\n"
        qa_spliter = "\n"
        colon = ""  # 这里不再需要冒号
        
        # 调用父类构造函数
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter, colon=colon)

        # 重写模板: 
        # 若存在 system_inst，则模板为：
        # <|im_start|>system
        # {system_inst}
        # <|im_end|>
        # <|im_start|>user
        # {prompt}
        # <|im_end|>
        # <|im_start|>assistant
        #
        # 若不存在 system_inst，则跳过 system 块，但这里我们默认有 system_inst。
        
        if self.system_inst is None:
            self.template = (
                self.role1 
                + "{prompt}\n"
                + "<|im_end|>\n"
                + self.role2
            )
        else:
            self.template = (
                "<|im_start|>system\n"
                + self.system_inst
                + "\n<|im_end|>\n"
                + self.role1
                + "{prompt}\n"
                + "<|im_end|>\n"
                + self.role2
            )

    def update_template(self, outputs, chunk_prefilling=0):
        # 对于 Qwen2 来说，我们通常不需要频繁更新模板。
        # 若有特殊需求，可在此根据逻辑微调。
        # 这里保持简单，不做改动：
        if chunk_prefilling:
            self.template = (
                self.role1
                + "{prompt}\n"
                + "<|im_end|>\n"
                + self.role2
            )
        else:
            # 若需要将对话上下文追加到模板中，可在此实现
            # 简单起见，不做复杂处理
            self.template = (
                "<|im_start|>system\n"
                + self.system_inst
                + "\n<|im_end|>\n"
                + self.role1
                + "{prompt}\n"
                + "<|im_end|>\n"
                + self.role2
            )

def get_prompter(model_type, model_path="", short_prompt=False, empty_prompt=False):
    if model_type.lower() == "qwen2":
        return Qwen2Prompter()
    else:
        raise ValueError(f"model type {model_type} is not supported")
