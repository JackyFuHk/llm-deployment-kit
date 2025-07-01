'''
用vllm(Qwen2.5)+langchain+llamaindex+fastapi+EasyOCR搭建RAG知识库，可检索详细合同。
数据源：1. 客户合同数据pdf 2. 产品知识pdf（文本，图片，表格）  3. 产品价格（excel）
'''
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM,AutoModel
from glob import glob
import pdfplumber
import torch
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
app = FastAPI()

# 转矢量模型用bge
embedding_model_name = "/mnt/f/ubuntu/deployment/model/bge-base-en-v1___5"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name,trust_remote_code=True,local_files_only=True)
model = AutoModel.from_pretrained(embedding_model_name)

# reranker
reranker_model_name = "/mnt/f/ubuntu/deployment/model/bge-reranker-base"
reranker_model = AutoModel.from_pretrained(embedding_model_name,trust_remote_code=True,local_files_only=True)

# vector store
client = QdrantClient(":memory:")  # 内存模式（测试用）
collection_name = "my_collection"
vector_size = 768  # BGE模型的向量维度（base模型为768，large为1024）

# 检查集合是否存在，不存在则创建
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE  # 使用余弦相似度
        )
    )

def get_embedding(text: str) -> list:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 取 [CLS] 位置的向量作为句子表示
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding

def get_sigle_embedding(text: str) -> list:
    return get_embedding([text])[0]

def init_rag():
    # 读取文本数据，切块，分为
    contract_datas = glob("./data/contract/*.pdf")
    invoice_datas = glob("./data/invoice/*.jpg")
    product_datas = glob("./data/product_info/*.pdf")

    # 处理合同数据，自动判断是哪种数据形式。
    for contract_data in contract_datas:
        # 合同数据，转文本，存数据库
        extract_pdf(contract_data,"contract")
    
    for invoice_data in invoice_datas:
        # 都是图片，要用OCR识别文字，转表格，转html，存入数据库。
        pass

    for product_data in product_datas:
        # 全部都是文字信息。基础按段落分块，
        extract_pdf(product_data,"product_info")

def extract_pdf(pdf_path,data_type):
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            chucks = text.split("\n")
            vectors = [get_embedding(chunk) for chunk in chucks]
            # 构造 Qdrant 的点（Point）结构
            points = [
                PointStruct(
                    id=idx,  # 唯一ID（可以是自增数字或UUID）
                    vector=vector,
                    payload={"text": text,"data_type":data_type}  # 附加元数据
                )
                for idx, (text, vector) in enumerate(zip(chucks, vectors))
            ]

            # 批量插入数据
            client.upsert(
                collection_name=collection_name,
                points=points
            )


            # table = page.extract_table()
            # print(table)






class LLM_Model:
    def __init__(self):
        # 语言模型
        self.model_path = "/mnt/f/ubuntu/deployment/model/Qwen2___5-1___5B-Instruct-GPTQ-Int4"
        self.language_model = AutoModelForCausalLM(
            self.model_path,
            device_map="auto",
            trust_remote_code=True, # 必须，允许加载qwen的自定义代码
            local_files_only=True, # 强制在本地加载
            quantize_config='gptq', # 必须，量化配置
        )
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=True,local_files_only=True) # 强制在本地加载tokenizer

    def generate_text(self, prompt, max_length=200):
        pass




init_rag()


query_text = "what material is used for the double wall paper cup?"
query_vector = get_embedding(query_text)

# 相似性搜索
hits = client.query_points(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=5,  # 返回前5个结果
    with_payload=True,  # 返回元数据
    with_vectors=False,  # 不返回向量（节省带宽）
)
# 打印结果
for hit in hits:
    print(f"ID: {hit.id}, 相似度: {hit.score:.4f}")
    print(f"文本: {hit.payload['text']}\n---")






# class Input(BaseModel):
#     prompt: str

# @app.post("/query/")
# async def chat(input: Input):
#     prompt = input.prompt
#     # 检索内容（元数据筛选）

#     # 语言模型生成文本

    
#     return {"response": "Hello, world!"}



# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)