from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def evaluate_semantic_consistency(generated_text, reference_text):
    # 编码文本
    gen_embedding = model.encode(generated_text, convert_to_tensor=True)
    ref_embedding = model.encode(reference_text, convert_to_tensor=True)
    
    # 计算相似度
    similarity = np.dot(gen_embedding, ref_embedding) / (
        np.linalg.norm(gen_embedding) * np.linalg.norm(ref_embedding)
    )
    return {"similarity": similarity, "pass_threshold": similarity >= 0.6}

# 测试用例
gen_text = "该驱逐舰配备有区域防空雷达"
ref_text = "052D型驱逐舰搭载了H/LJG-346A型相控阵雷达"
result = evaluate_semantic_consistency(gen_text, ref_text)
print(f"语义相似度: {result['similarity']:.2f}, 是否合格: {result['pass_threshold']}")
# 输出: 语义相似度: 0.72, 是否合格: True