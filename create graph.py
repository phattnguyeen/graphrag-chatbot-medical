from neo4j import GraphDatabase
import pandas as pd

# Cấu hình kết nối
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"

driver = GraphDatabase.driver(uri, auth=(username, password))

# Load dữ liệu
df = pd.read_csv("vinmec_medical_graph.csv")

# Tạo các node + quan hệ
def create_full_graph(tx, disease, overview, causes, symptoms, treatments, prevention, diagnoses):
    # Disease node
    tx.run("""
        MERGE (d:Disease {name: $disease})
        SET d.overview = $overview
    """, disease=disease, overview=overview)

    for cause in causes:
        if cause:
            tx.run("""
                MERGE (c:Cause {name: $cause})
                WITH c
                MATCH (d:Disease {name: $disease})
                MERGE (d)-[:DISEASE_CAUSE]->(c)
            """, cause=cause.strip(), disease=disease)

    for symptom in symptoms:
        if symptom:
            tx.run("""
                MERGE (s:Symptom {name: $symptom})
                WITH s
                MATCH (d:Disease {name: $disease})
                MERGE (d)-[:HAS_SYMPTOM]->(s)
            """, symptom=symptom.strip(), disease=disease)

    for treatment in treatments:
        if treatment:
            tx.run("""
                MERGE (t:Treatment {name: $treatment})
                WITH t
                MATCH (d:Disease {name: $disease})
                MERGE (d)-[:DISEASE_CUREWAY]->(t)
            """, treatment=treatment.strip(), disease=disease)

    for prev in prevention:
        if prev:
            tx.run("""
                MERGE (p:Prevention {name: $prev})
                WITH p
                MATCH (d:Disease {name: $disease})
                MERGE (d)-[:DISEASE_PREVENT]->(p)
            """, prev=prev.strip(), disease=disease)

    for diag in diagnoses:
        if diag:
            tx.run("""
                MERGE (dm:DiagnoseMethod {name: $diag})
                WITH dm
                MATCH (d:Disease {name: $disease})
                MERGE (d)-[:DIAGNOSE_BY]->(dm)
            """, diag=diag.strip(), disease=disease)

# Tạo quan hệ ACCOMPANY_WITH giữa các symptom cùng bệnh
def create_accompany_with(tx):
    tx.run("""
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s1:Symptom),
              (d)-[:HAS_SYMPTOM]->(s2:Symptom)
        WHERE id(s1) < id(s2)
        MERGE (s1)-[:ACCOMPANY_WITH]->(s2)
    """)

# Chạy import
with driver.session() as session:
    for _, row in df.iterrows():
        disease = row["disease"]
        overview = row["overview"] if not pd.isna(row["overview"]) else ""
        causes = str(row["causes"]).split("\n") if not pd.isna(row["causes"]) else []
        symptoms = str(row["symptoms"]).split("\n") if not pd.isna(row["symptoms"]) else []
        treatments = str(row["treatments"]).split("\n") if not pd.isna(row["treatments"]) else []
        prevention = str(row["prevention"]).split("\n") if not pd.isna(row["prevention"]) else []
        diagnoses = str(row["diagnose"]).split("\n") if not pd.isna(row["diagnose"]) else []

        session.write_transaction(create_full_graph, disease, overview, causes, symptoms, treatments, prevention, diagnoses)

    session.write_transaction(create_accompany_with)

driver.close()
print("Save graph successfully")
