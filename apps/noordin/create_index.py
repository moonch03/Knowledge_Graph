import os
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

# Configuration
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PASSWORD or not OPENAI_API_KEY:
    raise ValueError("NEO4J_PASSWORD and OPENAI_API_KEY must be set in environment variables")

AUTH = ("neo4j", PASSWORD)
driver = GraphDatabase.driver(URI, auth=AUTH)
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------
# Terror Network Vector Embeddings
# --------------------
def create_terror_vector_embeddings():
    """Generates and stores vector embeddings for Person nodes in the Noordin database."""
    with driver.session(database="noordin") as session:
        print("🔍 Reading Noordin network data...")

        result = session.run("""
            MATCH (p:Person)-[r]->(q:Person)
            RETURN
              p.id AS pid,
              p.name AS pname,
              type(r) AS rel,
              q.name AS qname,
              r.observed AS observed,
              r.weight AS weight
        """)

        # Person별 관계 설명 누적
        person_docs = defaultdict(list)

        for record in result:
            sentence = (
                f"{record['pname']} is connected to {record['qname']} "
                f"via {record['rel'].lower().replace('_', ' ')} relationship. "
                f"Observed: {record['observed']}, Weight: {record['weight']}."
            )
            person_docs[record["pid"]].append(sentence)

        print(f"🧠 임베딩 대상 인물 수: {len(person_docs)}")

        # --------------------
        # 임베딩 생성 & 저장
        # --------------------
        for pid, sentences in person_docs.items():
            text_to_embed = " ".join(sentences)

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text_to_embed
            )
            embedding = response.data[0].embedding

            session.run("""
                MATCH (p:Person {id: $id})
                SET p.embedding = $embedding
            """, id=pid, embedding=embedding)

            print(f"✅ {pid} 벡터 저장 완료")

        # --------------------
        # Vector Index 생성
        # --------------------
        print("📐 벡터 인덱스 생성 중...")

        session.run("DROP INDEX person_vector_index IF EXISTS")
        session.run("""
            CREATE VECTOR INDEX person_vector_index IF NOT EXISTS
            FOR (p:Person) ON (p.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            }
        """)

        print("🚀 'person_vector_index' 생성 완료!")

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    create_terror_vector_embeddings()
    driver.close()
