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



class RagSystem:
    def __init__(self,embedding_model_name = "/mnt/f/ubuntu/deployment/model/bge-base-en-v1___5",reranker_model_name = "/mnt/f/ubuntu/deployment/model/bge-reranker-base"):
        # emebdding model use bge
        self.embedding_model_name = embedding_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name,trust_remote_code=True,local_files_only=True)
        self.model = AutoModel.from_pretrained(self.embedding_model_name)

        # bge reranker 
        self.reranker_model_name = reranker_model_name
        self.reranker_model = AutoModel.from_pretrained(self.embedding_model_name,trust_remote_code=True,local_files_only=True)

        # vector store
        self.client = QdrantClient(":memory:")  # test mode, user memory mode for real use
        self.collection_name = "my_collection"
        self.vector_size = 768  # bge embedding size nomally 768 or 1024

        # check collection and create if not exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE  # cosine distance
                )
            )
    
    def read_pdf(self,pdf_path):
        text_list = []
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.extract_text()
            text_list.append(text)
        return text_list
                

    def chunk_doc(self,text:str,max_chunk_size=512):
        # chunk document into small chucks for indexing by punctuations
        chucks = text.split("\n")
        chucks = [chunk for chunk in chucks if len(chunk) > 0]
        chucks = [chunk for chunk in chucks if len(chunk) <= max_chunk_size]
        return chucks

    def get_embedding(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # get [CLS] position vector as sentence representation
        outputs = self.model(**inputs)
        # 取 [CLS] 位置的向量作为句子表示
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        return embedding



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