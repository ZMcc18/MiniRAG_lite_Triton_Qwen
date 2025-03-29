import torch
import logging
import gc
from typing import List

logger = logging.getLogger(__name__)

def get_dtype_size(dtype: torch.dtype) -> int:
    """获取张量数据类型的字节大小，如torch.float32 → 4 字节"""
    return torch.tensor([], dtype=dtype).element_size()

class ComputeMaxAvailableBlocks:
    """
        - 通过执行虚拟输入的前向传播来分析模型的内存使用情况
        - 计算在剩余空闲内存中可以分配的最大GPU块数量
        这个类主要用于大语言模型推理时的内存优化管理，
        特别是在使用KV缓存(KV Cache)时精确计算可用的内存空间。
        或者可以构建cuda graph优化之后再调用内存计算，cuda graph会改变内存分配模式
        通常会减少内存碎片，优化显存使用。
        并且如果使用cuda graph，计算内存空间就一定得在后面算了，因为cuda graph会改变内存分配模式。
    """
    def __init__(
        self, 
        num_layers, 
        hidden_size, 
        num_heads, 
        num_kv_heads, 
        head_dim = None, 
        gpu_memory_utilization=0.9, 
        block_size=1, 
        dtype="float16"
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.head_dim = head_dim

        self.gpu_memory_utilization = gpu_memory_utilization
        self.block_size = block_size # 一个 block 表示多少个 tokens
        self.dtype = dtype
        
        if self.dtype in ["float16", "bfloat16", "fp16", "bfp16"]:
            self.dtype_size = 2
        elif self.dtype in ["int8", "fp8"]:
            self.dtype_size = 1 # byte
        else:
            print(f"Unsupported dtype: {self.dtype_size}!")

    def compute_cache_block_size_bytes(self): 
        """
        获取一个KV缓存块大小的字节数
        根据模型参数计算每个token的KV缓存大小
        每个token的KV缓存大小 = num_kv_heads * head_dim * 2 * dtype_size
        然后由于每个token存在于多个layer中，所以需要乘以num_layers
        最后乘以block_size，表示一个block包含多少个tokens
        因此，一个KV缓存块的大小 = num_kv_heads * head_dim * 2 * dtype_size * num_layers * block_size
        """
        if self.head_dim is None:
            head_size = self.hidden_size // self.num_heads
        else:
            head_size = self.head_dim
        
        num_layers = self.num_layers
        num_kv_heads = self.num_kv_heads
        # num_heads * head_size = hidden_size
        kv_cache_token_bytes_per_layer = (num_kv_heads * head_size) * 2 * self.dtype_size
        transformer_kv_cache_token_bytes = kv_cache_token_bytes_per_layer * num_layers

        transformer_kv_cache_blocks_bytes = transformer_kv_cache_token_bytes * self.block_size

        return transformer_kv_cache_blocks_bytes
    
    def compute_num_available_blocks(self, model_path=None, dummy_input = None, model_byes=None):
        """
        评估模型的峰值内存使用情况，以确定在不发生内存溢出的情况下可以分配的 KV（键值）缓存块的数量。

        该方法首先清理 CUDA 缓存，然后使用虚拟输入执行一次前向传播，以评估模型的内存使用情况。
        接着，计算在剩余可用内存下，最多可以分配的 GPU 和 CPU 缓存块数量。

        提示：
            可以通过调整 `gpu_memory_utilization` 参数来限制 GPU 内存的使用。
        """
        # 清理 CUDA 缓存，以确保获取准确的内存使用信息
        # NOTE: torch.cuda.empty_cache() 用于释放 GPU 上由缓存分配器持有的未占用内存。
        # NOTE: torch.cuda.reset_peak_memory_stats() 用于重置 CUDA 内存分配器所跟踪的“峰值”统计数据。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 获取当前 GPU 的空闲内存和总内存（单位：字节）
        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()

        # TODO: 使用虚拟输入执行一次前向传播，以评估模型的内存使用情况

        # 同步 CUDA 操作，确保内存信息准确
        torch.cuda.synchronize()
        # 计算模型加载后的峰值内存使用量. Get the peak memory allocation recorded by torch
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        
        # 清理未使用的缓存，计算非 Torch 分配的内存. 检查是否有任何剩余内存可能已在“torch”之外的 gpu 上分配。例如，NCCL通信库 操作在前向传递期间可能会使用几 GB
        torch.cuda.empty_cache()
        torch_allocated_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 获取当前PyTorch实际占用的显存
        
        total_allocated_bytes = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0] 
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes # 得到了非torch占用的内存
        
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations

        available_kv_cache_memory = (   # 总gpu内存*提前说好的上限%，然后减去模型峰值占用，就是可以用来分配kv cache的内存
            # 计算可用的 GPU 内存，考虑了 GPU 内存利用率和非 Torch 分配的内存
            # 公式：可用内存 = 总 GPU 内存 * GPU 内存利用率 - 峰值内存 - 非 Torch 分配的内存
            # 这个公式假设我们希望在不超过 GPU 内存上限的情况下，最大化可用内存。
            total_gpu_memory * self.gpu_memory_utilization -
            peak_memory)
        
        # 计算每个缓存块的大小
        cache_block_size = self.compute_cache_block_size_bytes()
        # 计算在剩余可用内存下，最多可以分配的 GPU 缓存块数量
        num_gpu_blocks = int(
            # 公式：可分配的 GPU 缓存块数量 = 可用内存 / 缓存块大小
            (total_gpu_memory * self.gpu_memory_utilization -
            peak_memory) // cache_block_size
        )
        # 确保缓存块数量不为负数
        num_gpu_blocks = max(num_gpu_blocks, 0)
        print("num_gpu_blocks:",type(num_gpu_blocks))
        print("Before logger.info")
        logger.info(
                " Memory profiling results: total_gpu_memory 总gpu显存 = %.2f GB \n"
                "    initial_memory_usage 初始可用显存  = %.2f GB peak_torch_memory 峰值显存占用 = %.2f GB \n"
                "    memory_usage_post_profile 总共可分配显存 = %.2f GB \n"
                "    non_torch_memory = %.2f GB, kv_cache_size = %.2f GB \n"
                "    gpu_memory_utilization 自己设置的显存可用比 = %.2f \n"
                "    num_gpu_blocks 显存块数量 = %d",
                total_gpu_memory / (1024**3),
                (total_gpu_memory - free_memory_pre_profile) / (1024**3),
                (peak_memory - non_torch_allocations) / (1024**3),
                total_allocated_bytes / (1024**3),
                non_torch_allocations / (1024**3),
                available_kv_cache_memory / (1024**3),
                self.gpu_memory_utilization,
                num_gpu_blocks,
                )

        # 进行垃圾回收，释放未使用的内存
        gc.collect()
        # 再次清理 CUDA 缓存
        torch.cuda.empty_cache()
        # 返回可分配的 GPU 和 CPU 缓存块数量（此处 CPU 块数量为 0）

        return num_gpu_blocks

class KVCacheMemoryManager:
    def __init__(self, num_layers, num_kv_heads, head_dim, gpu_num_blocks, block_size=1, dtype=torch.float16, device="cuda"):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gpu_num_blocks = gpu_num_blocks # 手动设定的给kv cache 内存管理分配的可用 blocks 数目:gpu_num_blocks
        self.block_size = block_size
        self.max_num_tokens = gpu_num_blocks * block_size

        self.dtype = dtype
        self.device = device
        self.can_use_mem_size = gpu_num_blocks # 可用的 kv cache tokens 数量

        # 定义 kv 内存位置索引和内存使用状态变量
        self.kv_mem_pos_indexs = torch.arange(0, self.max_num_tokens, dtype=torch.long, device="cuda")
        self.kv_mem_use_state = torch.zeros(self.max_num_tokens, dtype = torch.int32, device="cuda")

        # Initialize the gpu_kv_buffer
        self.init_kv_buffers(
            self.max_num_tokens,
            head_dim, num_kv_heads, num_layers, 
            dtype, device)

    def init_kv_buffers(self, 
        max_num_tokens,
        head_dim, num_kv_heads, num_layers,
        dtype,
        device: str="cuda"
    )-> List[torch.Tensor]:
        # 用于初始化KV缓存(KV Cache)的GPU缓冲区
        # kv cache shape: config.max_batch_size, config.max_seq_len, self.num_kv_heads, self.head_dim
        # max_num_tokens = max_num_blocks * self.block_size
        # TODO 修改 kv buffer 形状支持 PagedAttention
        self.gpu_kv_buffer = [
            torch.empty((max_num_tokens, 2 * num_kv_heads, head_dim), dtype=dtype, device=device) for _ in range(num_layers)
        ]
        logger.debug(f"gpu_kv_buffer per layer shape: {self.gpu_kv_buffer[0].shape}")

    @torch.no_grad()
    # 动态分配kvcache空间，完成了token_attention
    def alloc_kvcache(self, need_size):
        if need_size > self.can_use_mem_size:
            logger.warning(f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}")
            return None
        
        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        select_index = can_use_pos_index[0:need_size]
        self.add_ref(select_index)
        
        return select_index    

    @torch.no_grad()
    # 分配连续的KV缓存空间
    def alloc_contiguous_kvcache(self, need_size):
        if need_size > self.can_use_mem_size:
            logger.warning(f"warn no enough contiguous cache need_size {need_size} left_size {self.can_use_mem_size}")
            return None

        # 获取未使用的内存块索引
        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        N = can_use_pos_index.numel()
        if N >= need_size:
            # 正确地计算 start_indexs 和 end_indexs. 
            # NOTE: 起始索引不能大于 N - need_size, 又因为 [: index] 切片操作是不包含 index 的, 所以需要将 N - need_size 加 1
            start_indexs = can_use_pos_index[:N - need_size + 1]
            # NOTE: can_use_pos_index[3:], 将获取索引为 3 到 9 的元素。
            end_indexs = can_use_pos_index[need_size - 1:]
            diff = end_indexs - start_indexs

            # 寻找连续的块，差值应为 need_size - 1
            contiguous_blocks = (diff == need_size - 1).nonzero(as_tuple=True)[0]

            if contiguous_blocks.numel() > 0:
                start_index = start_indexs[contiguous_blocks[0]].item()
                end_index = start_index + need_size
                select_index = self.kv_mem_pos_indexs[start_index:end_index]
                self.add_ref(select_index)
                return select_index, start_index, end_index

        return None
        
    @torch.no_grad()
    def alloc_kvcache_index(self, need_size):
        alloc_mem = self.alloc_contiguous_kvcache(need_size)
        if alloc_mem is not None:
            select_index, start_index, end_index = alloc_mem
            kv_cache = None
        else:
            select_index = self.alloc_kvcache(need_size)
            kv_cache = torch.empty(
                (need_size, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        
        return select_index.to(torch.int32), kv_cache

    # 增加引用计数
    @torch.no_grad()
    def add_ref(self, token_index: torch.Tensor):
        state = self.kv_mem_use_state[token_index]
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size -= all_tokens - has_used_tokens
  
        self.kv_mem_use_state[token_index] += 1
        return

    # 减少引用计数
    @torch.no_grad()
    def release_ref(self, token_index: torch.Tensor):
        # 使用 unique 方法获取 token_index 中唯一的 token 索引，并返回每个唯一索引在原始张量中出现的次数。
        token_index, counts = token_index.unique(return_counts=True)
        # 当引用计数减少到零时，意味着该缓存块可以被释放或重新分配。
        self.kv_mem_use_state[token_index] -= counts
        state = self.kv_mem_use_state[token_index]
        used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size += all_tokens - used_tokens
        return

    # 释放键值缓存缓冲区
    def _free_buffers(self):
        self.gpu_kv_buffer = None
    
    # 释放指定的kv cache 内存块索引
    @torch.no_grad()
    def free(self, free_index):
        free_index = free_index.long()
        self.release_ref(free_index)
        if self.can_use_mem_size == len(self.mem_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return
    
    # 释放所有内存
    @torch.no_grad()
    def free_all(self,):
        self.can_use_mem_size = len(self.kv_mem_use_state)
        self.kv_mem_use_state[:] = 0