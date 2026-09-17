# ==========================================
# 前文保留：boardgame.py 的全文代码（切片、向量匹配、Faiss、Rerank）
# 到 print(f"最匹配的规则: {final_answer}") 为止
# ==========================================
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
def sliding_window_chunking(text, chunk_size=200, overlap=80):
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

tokenized_corpus = [list(jieba.cut(sentence)) for sentence in sentences]
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 关键词索引构建完成！")


question = "每轮第二多的颜色值多少筹码"

with torch.inference_mode():
    inputs = tokenizer(question, return_tensors="pt", padding=True, max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 提取问题向量
    vector = dual_model(**inputs)[1]
    q_vector = vector.cpu().numpy()

print("问题向量维度:", q_vector.shape)

faiss.normalize_L2(q_vector)
scores, indexes = index.search(q_vector, 20)


tokenized_query = list(jieba.cut(question))
bm25_scores = bm25.get_scores(tokenized_query)
# np.argsort 返回从小到大的索引，[::-1] 倒序取最大的前 20 个
bm25_candidates_idx = np.argsort(bm25_scores)[::-1][:20].tolist()

# 提取召回的 Top-10 规则内容
#topk_result = data.values[indexes[0].tolist()]
# 提取 FAISS 找出来的行号索引（整数列表），而不是直接提取文本
faiss_candidates_idx = indexes[0].tolist()
#print("召回的规则候选：\n", topk_result[:, 0])
merged_indexes = list(set(faiss_candidates_idx + bm25_candidates_idx))
candidate_texts = [sentences[i] for i in merged_indexes]

print(f"双路召回完毕，FAISS与BM25去重后共获得 {len(candidate_texts)} 个候选切片。")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 1. 加载模型与 Tokenizer
reranker_name = "BAAI/bge-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(reranker_name)
cross_model = AutoModelForSequenceClassification.from_pretrained(reranker_name).to(device)
cross_model.eval()

# 2. 准备候选文本和问题
candidate = candidate_texts
ques = [question] * len(candidate)

# 3. 将问题和候选规则拼接: [CLS] question [SEP] candidate [SEP]
inputs = tokenizer(ques, candidate, return_tensors="pt", padding=True, max_length=512, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 4. 执行推理重排
with torch.inference_mode():
    # 获取 logits 并展平为一维张量
    scores = cross_model(**inputs).logits.view(-1)
    # 选取打分最高的候选规则索引
    top_3_idx = torch.argsort(scores, descending=True)[:3]
    top_3_answers = [candidate_texts[i] for i in top_3_idx]
    final_context = "\n---\n".join(top_3_answers)

# 5. 输出最优匹配的规则答案
#final_answer = candidate[best_idx]
#print(f"用户问题: {question}")
#print(f"最匹配的规则: {final_answer}")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("\n开始加载生成大模型 (LLM)...")

# 1. 选择并加载生成式大模型 (此处以轻量级的 Qwen2.5-1.5B-Instruct 为例)
# 如果你显存充裕，可以替换为 "Qwen/Qwen2.5-7B-Instruct" 等更大参数的模型
llm_name = "Qwen/Qwen2.5-1.5B-Instruct" 

llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16,  # 使用半精度加载，节省显存
    device_map="auto"           # 自动分配到可用的 GPU 上
)
llm_model.eval()

# 2. 构建 RAG 专属提示词模板 (Prompt Template)
# 【关键设计】：必须用强指令约束大模型“严禁胡编乱造”，只能基于检索到的 Context 回答。
prompt_template = """你是一个专业的桌游规则裁判。请严格根据下面提供的【规则原文】来回答玩家的【问题】。
如果你发现【规则原文】中没有包含答案，或者原文与问题无关，请直接回答：“根据当前的规则资料，我无法找到关于这个问题的确切说明。”，请勿凭空捏造。

【规则原文】：
{context}

【玩家问题】：
{question}

请给出清晰、准确、自然的自然语言解答：
"""

# 3. 将检索结果与问题填入模板 (Augmentation)
# final_answer 和 question 变量来自前文的重排结果
prompt = prompt_template.format(context=final_context, question=question)

# 4. 适配主流大模型的对话格式 (Chat Template)
messages = [
    {"role": "system", "content": "你是一个严谨且乐于助人的桌游规则客服。"},
    {"role": "user", "content": prompt}
]

text = llm_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

llm_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

print("\n生成模型准备完毕，正在思考答案...\n" + "="*50)

# 5. 执行文本生成 (Generation)
with torch.inference_mode():
    generated_ids = llm_model.generate(
        **llm_inputs,
        max_new_tokens=256,    # 限制生成的最大长度
        temperature=0.1,       # RAG场景下温度设置得很低(0.1)，确保回答的客观性和事实性
        do_sample=True,
        top_p=0.8
    )

# 6. 后处理：剥离掉输入的 Prompt，只提取大模型新生成的回复文本
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(llm_inputs.input_ids, generated_ids)
]
final_response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 7. 最终展示
print(f"【玩家提问】: {question}")
#print(f"【检索依据】: (由 Faiss & BGE-Reranker 提供)\n{final_context}")
print("-" * 50)
print(f"【AI 裁判解答】:\n{final_response}")