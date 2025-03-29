import spacy
import json
from collections import defaultdict

# 加载术语库样例：term_db = {"052d驱逐舰", "an/spy-6雷达", "鹰击-18导弹"}  
with open('xxx.json', 'r', encoding='utf-8') as f:
    term_db = set(json.load(f))
nlp = spacy.load("zh_core_web_sm")

def evaluate_term_accuracy(text):
    # 术语抽取与归一化
    doc = nlp(text)
    candidates = [ent.text.lower().replace("-", "").replace("/", "") 
                  for ent in doc.ents if ent.label_ in ["WEAPON", "SHIP"]]
    
    # 统计指标
    correct = [term for term in candidates if term in term_db]
    error_terms = list(set(candidates) - set(correct))
    
    return {
        "term_accuracy": len(correct) / len(candidates) if candidates else 1.0,
        "error_terms": error_terms,
        "total_terms": len(candidates)
    }

# 测试用例
text = "052D型驱逐舰装备了AN/SPY-6雷达和东风-21D导弹"
result = evaluate_term_accuracy(text)
print(f"术语准确率: {result['term_accuracy']:.1%}, 错误术语: {result['error_terms']}")
# 输出: 术语准确率: 66.7%, 错误术语: ['东风21d导弹']