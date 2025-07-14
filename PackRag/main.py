'''
用vllm(Qwen2.5)+langchain+llamaindex+fastapi+EasyOCR搭建RAG知识库，可检索详细合同。
数据源：1. 客户合同数据pdf 2. 产品知识pdf（文本，图片，表格）  3. 产品价格（excel）
'''
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification ,AutoModel
from glob import glob
from pypdf import PdfReader
import torch
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class RagSystem:
    def __init__(self,embedding_model_name = "/mnt/f/ubuntu/deployment/model/bge-base-en-v1___5",reranker_model_name = "/mnt/f/ubuntu/deployment/model/bge-reranker-base"):
        # emebdding model use bge
        self.embedding_model_name = embedding_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name,trust_remote_code=True,local_files_only=True)
        self.model = AutoModel.from_pretrained(self.embedding_model_name)

        # bge reranker 
        self.reranker_model_name = reranker_model_name
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name,trust_remote_code=True,local_files_only=True)

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
        reader = PdfReader(pdf_path)
        number_of_pages = len(reader.pages)
        for i in range(number_of_pages):
            page = reader.pages[0]
            text_list.append(page.extract_text())
        return text_list
                
    def chunk_doc(self,text:str,max_chunk_size=512):
        # chunk document into small chucks for indexing by punctuations
        chucks = text.split("\n")
        chucks = [chunk for chunk in chucks if len(chunk) > 0]
        chucks = [chunk for chunk in chucks if len(chunk) <= max_chunk_size]
        return chucks

    def get_embedding(self, text: str) -> list:
        inputs = self.tokenizer(text, padding=True, return_tensors="pt", truncation=True, max_length=512)
        # get [CLS] position vector as sentence representation
        outputs = self.model(**inputs)
        # 取 [CLS] 位置的向量作为句子表示
        embedding = outputs.last_hidden_state[:, 0, :].mean(dim=0).tolist()
        return embedding

    def save_e(self, points: list):
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )   
    
    def search_e(self, query: str, top_k: int = 10) -> list:
        # search embedding in vector store
        query_embedding = self.get_embedding(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return results

    def rerank(self, query: str, doc_ids: list) -> list:
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=doc_ids,
            with_payload=True,  # 返回 payload（文档内容）
            with_vectors=False,  # 不需要向量
        )

        # 转换为标准格式
        documents = []
        for record in records:
            print(record)
            documents.append({
                "id": str(record.id),  # 转回字符串（如果需要）
                "text": record.payload.get("text", ""),
                "metadata": record.payload.get("metadata", {}),
            })
   
        if not documents:
            return []
        # (query, doc_text)
        inputs = self.tokenizer(
            [(query, doc["text"]) for doc in documents],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # get scores for each document
        with torch.no_grad():
            outputs = self.reranker_model(**inputs)
            scores = outputs.logits.squeeze().tolist()  # [score1, score2, ...]
        if isinstance(scores, float):
            scores = [scores]
        sorted_documents = [
            doc for _, doc in sorted(
                zip(scores, documents),
                key=lambda x: x[0],  # 排序 key 是分数
                reverse=True
            )
        ]

        # 提取排序后的 doc_id
        sorted_doc_ids = [doc["id"] for doc in sorted_documents]

        return sorted_doc_ids

    def search(self, query: str, top_k: int = 10) -> list:
        # search documents by embedding and reranking
        results = self.search_e(query, top_k)
        sorted_doc_ids = self.rerank(query, [dict(doc)['id'] for doc in results])
        return sorted_doc_ids[:top_k]


if __name__ == '__main__':
    my_rag_system = RagSystem()
    # read pdf file
    pdf_path = "/mnt/f/ubuntu/deployment/PackRag/dataset/double wall paper cup detail.pdf"
    # chunk document into small chucks for indexing by punctuations
    chunks_list = []
    text_list = my_rag_system.read_pdf(pdf_path)
    for text in text_list:
        chunks = my_rag_system.chunk_doc(text)
        chunks_list.append(chunks)
    # save embedding to vector store
    points = []
    
    for i, chunks in enumerate(chunks_list):
        doc_id = i
        points.append(
        PointStruct(
            id=doc_id,
            vector = my_rag_system.get_embedding(chunks),
            payload = {"product_name":"Double wall paper cup","text":chunks}
        ))
    my_rag_system.save_e(points)
    
    while True:
        query = input("input query: ")
        results = my_rag_system.search(query, top_k = 10)
        print(results)
   
