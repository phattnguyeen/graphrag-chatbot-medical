import os
import torch
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
from underthesea import word_tokenize

# ==== 1. Hàm tạo embedding ====

def hf_text_embedding(text, tokenizer, model):
    """Sinh embedding từ text bằng HuggingFace model"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.squeeze().cpu().numpy().tolist()

# ==== 2. Class quản lý Vector Index ====

class Neo4jVectorManager:
    def __init__(self, neo4j_config, model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base", embedding_dim=768):
        """Khởi tạo kết nối Neo4j và load model"""
        self.driver = GraphDatabase.driver(
            neo4j_config["url"],
            auth=(neo4j_config["username"], neo4j_config["password"])
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim

    def generate_embeddings(self, label, text_field="name"):
        """Sinh embedding cho tất cả node theo label"""
        with self.driver.session() as session:
            print(f"🔵 Đang tạo embedding cho {label}...")
            results = session.run(f"""
                MATCH (n:{label})
                WHERE n.{text_field} IS NOT NULL
                RETURN id(n) AS id, n.{text_field} AS text
            """)
            for record in results:
                node_id = record["id"]
                text = record["text"]
                if text:
                    segmented = word_tokenize(text, format="text")
                    embedding = hf_text_embedding(segmented, self.tokenizer, self.model)
                    session.run("""
                        MATCH (n) WHERE id(n) = $id
                        SET n.embedding = $embedding
                    """, {"id": node_id, "embedding": embedding})

    def create_vector_index(self, label):
        """Tạo VECTOR INDEX cho label"""
        index_name = f"index_{label.lower()}"
        print(f"🛠️  Đang tạo vector index cho {label} ({index_name})...")
        with self.driver.session() as session:
            session.run(f"""
                CALL db.index.vector.createNodeIndex(
                    '{index_name}',
                    '{label}',
                    'embedding',
                    {self.embedding_dim},
                    'cosine'
                )
            """)
        print(f"✅ Đã tạo vector index {index_name}!")

    def process_all_labels(self, labels):
        """Tạo embedding và index cho nhiều label"""
        for label in labels:
            try:
                self.generate_embeddings(label)
                self.create_vector_index(label)
            except Exception as e:
                print(f"❌ Lỗi khi xử lý label {label}: {str(e)}")

    def similarity_search_git(self, label, query_text, top_k=5):
        """Tìm kiếm similarity trong vector index"""
        if label.lower() == "disease":
            top_k = 1
        segmented = word_tokenize(query_text, format="text")
        query_embedding = hf_text_embedding(segmented, self.tokenizer, self.model)
        index_name = f"index_{label.lower()}"

        with self.driver.session() as session:
            results = session.run(f"""
                CALL db.index.vector.queryNodes(
                    $index_name,
                    $top_k,
                    $embedding
                )
                YIELD node, score
                RETURN 
                labels(node)[0] AS label,
                CASE 
                    WHEN labels(node)[0] = 'Disease' THEN coalesce(node.overview, node.name)
                    ELSE node.name
                END AS text,score
                ORDER BY score DESC
            """, {
                "index_name": index_name,
                "top_k": top_k,
                "embedding": query_embedding
            })
            return [(record["label"], record["text"], record["score"]) for record in results]

    def search_across_labels(self, query_text, labels, top_k=3):
        """Tìm kiếm semantic trên nhiều labels"""
        all_results = []
        for label in labels:
            try:
                res = self.similarity_search_git(label, query_text, top_k=top_k)
                all_results.extend(res)
            except Exception as e:
                print(f"❗ Lỗi khi search label {label}: {str(e)}")
        all_results.sort(key=lambda x: x[2], reverse=True)  # Sort theo score
        return all_results

    def detect_label_from_query_git(self, query_text):
        """Tự động nhận diện label từ câu hỏi"""
        query = query_text.lower()
        if any(k in query for k in ["phòng", "ngừa", "tránh"]):
            return "Prevention"
        elif any(k in query for k in ["triệu chứng", "dấu hiệu", "biểu hiện"]):
            return "Symptom"
        elif any(k in query for k in ["nguyên nhân", "tại sao", "vì sao"]):
            return "Cause"
        elif any(k in query for k in ["chẩn đoán", "xét nghiệm"]):
            return "DiagnoseMethod"
        elif any(k in query for k in ["điều trị", "thuốc", "cách chữa", "phác đồ"]):
            return "Treatment"
        elif any(k in query for k in ["bệnh", "là gì", "thông tin", "định nghĩa"]):
            return "Disease"
        else:
            return "Disease"  # fallback

    def close(self):
        """Đóng kết nối Neo4j"""
        self.driver.close()
