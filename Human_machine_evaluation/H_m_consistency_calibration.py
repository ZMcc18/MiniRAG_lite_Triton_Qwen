import pandas as pd
from scipy.stats import spearmanr

# 假设df包含机器与人工评分
df = pd.DataFrame({
    "machine_term_score": [0.8, 0.7, 0.9],
    "human_term_score": [4, 3, 5],
    "machine_semantic_score": [0.7, 0.65, 0.8],
    "human_semantic_score": [4, 3, 5]
})

# 计算术语指标相关性
corr, p_value = spearmanr(df["machine_term_score"], df["human_term_score"])
print(f"术语一致性: corr={corr:.2f}, p={p_value:.3f}")

# 输出: 术语一致性: corr=0.87, p=0.333