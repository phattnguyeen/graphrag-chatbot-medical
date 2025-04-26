import os
import json
from underthesea import word_tokenize
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
from elasticsearch import Elasticsearch
import torch

def hf_text_embedding(text, tokenizer, model):
    """Hàm tự động tạo embedding vector từ text tiếng Việt"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.squeeze().cpu().numpy()

class HybridVectorSearch:
    def __init__(self, neo4j_config, elastic_config, model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"):
        # Kết nối Neo4j
        self.driver = GraphDatabase.driver(
            neo4j_config["url"],
            auth=(neo4j_config["username"], neo4j_config["password"])
        )

        # Kết nối Elastic
        self.elastic = Elasticsearch(**elastic_config)

        # Load PhoBERT SimCSE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def index_neo4j_nodes_to_elastic(self, label, index_name):
        """Lấy dữ liệu từ Neo4j, tạo embedding, lưu vào Elastic"""
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{label})
            RETURN id(n) AS id, n.name AS name
            """
            results = session.run(query)

            docs = []
            for r in results:
                text = r["name"]
                if text:
                    segmented = word_tokenize(text, format="text")
                    vector = hf_text_embedding(segmented, self.tokenizer, self.model)
                    docs.append({
                        "neo4j_id": r["id"],
                        "text": text,
                        "vector": vector
                    })

        # Nếu index đã tồn tại thì xóa
        if self.elastic.indices.exists(index=index_name):
            self.elastic.indices.delete(index=index_name)

        # Tạo lại index
        self.elastic.indices.create(index=index_name, mappings={
            "properties": {
                "text": {"type": "text"},
                "neo4j_id": {"type": "long"},
                "vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"}
            }
        })

        # Insert documents
        for i, doc in enumerate(docs):
            self.elastic.index(index=index_name, id=i, document=doc)

        print(f"✅ Đã index {len(docs)} node {label} vào ElasticSearch!")

    def search(self, query, index_name="cause_index", top_k=5):
        """Search văn bản từ Elastic và mapping về Neo4j"""
        seg_query = word_tokenize(query, format="text")
        q_vector = hf_text_embedding(seg_query, self.tokenizer, self.model)

        result = self.elastic.search(
            index=index_name,
            knn={
                "field": "vector",
                "query_vector": q_vector.tolist(),
                "k": top_k,
                "num_candidates": 500
            },
            _source=["neo4j_id", "text"]
        )

        neo4j_ids = [hit["_source"]["neo4j_id"] for hit in result["hits"]["hits"]]
        return self.fetch_neo4j_details(neo4j_ids)

    def fetch_neo4j_details(self, node_ids):
        """Mapping ID search được về nội dung node trong Neo4j"""
        details = []
        with self.driver.session() as session:
            for node_id in node_ids:
                res = session.run(
                    "MATCH (n) WHERE id(n) = $id RETURN labels(n) AS label, n.name AS name",
                    {"id": node_id}
                )
                for record in res:
                    details.append({
                        "label": record["label"],
                        "name": record["name"]
                    })
        return details
