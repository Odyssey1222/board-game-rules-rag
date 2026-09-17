import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import faiss

def sliding_window_chunking(text, chunk_size=200, overlap=50):
    """
    对长文本进行带重叠的滑动窗口切片
    
    参数:
        text (str): 需要切分的完整规则文本
        chunk_size (int): 每个文本块（Chunk）的最大字符长度
        overlap (int): 相邻文本块之间重叠的字符数量
    
    返回:
        list: 切分好的文本块列表
    """
    chunks = []
    text_length = len(text)
    
    # 步长 = 块长度 - 重叠长度
    step = chunk_size - overlap
    
    # 防止步长小于等于0导致死循环
    if step <= 0:
        raise ValueError("overlap 必须小于 chunk_size")
        
    for start in range(0, text_length, step):
        end = start + chunk_size
        chunk = text[start:end]
        
        # 过滤掉全是空白符的无效段落
        if chunk.strip():
            chunks.append(chunk.strip())
            
        # 如果已经覆盖到文本末尾，则提前结束
        if end >= text_length:
            break
            
    return chunks
# 读取桌游规则文本，并按换行符切分为段落（过滤掉空行）

# 假设我们读取了你前面提供的《现代艺术》规则文本
with open("./modern_art_rules.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# 将全文本按照 250字/块，50字重叠 进行切分
rule_chunks = sliding_window_chunking(full_text, chunk_size=250, overlap=50)

print(f"总共切分出 {len(rule_chunks)} 个带重叠的规则块。\n")

# 构建 DataFrame，模仿原有的 FAQ 数据结构
data = pd.DataFrame({"text": rule_chunks})
#print(data.head())


# 使用 HuggingFace 上专门用于文本检索的预训练 BERT 模型
model_name = "shibing624/text2vec-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 替代原文件中的 DualModel
dual_model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dual_model = dual_model.to(device)
dual_model.eval()

print("匹配模型加载成功！")
sentences = data["text"].to_list()
vectors = []

with torch.inference_mode():
    for i in tqdm(range(0, len(sentences), 32)):
        batch_sens = sentences[i: i + 32]
        inputs = tokenizer(batch_sens, return_tensors="pt", padding=True, max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # 获取 BERT 的 pooler_output 作为句向量 (等同于原文件的 dual_model.bert(**inputs)[1])
        vector = dual_model(**inputs)[1] 
        vectors.append(vector)

vectors = torch.concat(vectors, dim=0).cpu().numpy()
print("向量矩阵维度:", vectors.shape)

# 构建 768 维度的余弦相似度索引
index = faiss.IndexFlatIP(768)
faiss.normalize_L2(vectors)
index.add(vectors)
print(index)

question = "游戏什么时候结束"

with torch.inference_mode():
    inputs = tokenizer(question, return_tensors="pt", padding=True, max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 提取问题向量
    vector = dual_model(**inputs)[1]
    q_vector = vector.cpu().numpy()

print("问题向量维度:", q_vector.shape)

faiss.normalize_L2(q_vector)
scores, indexes = index.search(q_vector, 10)

# 提取召回的 Top-10 规则内容
topk_result = data.values[indexes[0].tolist()]
#print("召回的规则候选：\n", topk_result[:, 0])

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 1. 加载模型与 Tokenizer
reranker_name = "BAAI/bge-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(reranker_name)
cross_model = AutoModelForSequenceClassification.from_pretrained(reranker_name).to(device)
cross_model.eval()

# 2. 准备候选文本和问题
candidate = topk_result[:, 0].tolist()
ques = [question] * len(candidate)

# 3. 将问题和候选规则拼接: [CLS] question [SEP] candidate [SEP]
inputs = tokenizer(ques, candidate, return_tensors="pt", padding=True, max_length=512, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 4. 执行推理重排
with torch.inference_mode():
    # 获取 logits 并展平为一维张量
    scores = cross_model(**inputs).logits.view(-1)
    # 选取打分最高的候选规则索引
    best_idx = torch.argmax(scores).item()

# 5. 输出最优匹配的规则答案
final_answer = candidate[best_idx]
print(f"用户问题: {question}")
print(f"最匹配的规则: {final_answer}")