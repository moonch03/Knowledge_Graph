import os
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PASSWORD or not OPENAI_API_KEY:
    raise ValueError("NEO4J_PASSWORD and OPENAI_API_KEY must be set in environment variables")

AUTH = ("neo4j", PASSWORD)
client = OpenAI(api_key=OPENAI_API_KEY)

driver = GraphDatabase.driver(URI, auth=AUTH)

def create_vector_embeddings():
    """Fetches Gang data, creates embeddings using OpenAI, and stores them in Neo4j."""
    # Ensure the 'montreal' database is used
    with driver.session(database="montreal") as session:
        print("📥 Loading Gang nodes and relationship data...")

        query = """
        MATCH (g:Gang)
        OPTIONAL MATCH (g)-[:ALLIED_WITH]->(a:Gang)
        OPTIONAL MATCH (g)-[:ENEMY_OF]->(e:Gang)
        OPTIONAL MATCH (g)-[:CROSS_TERRITORY]->(c:Gang)
        RETURN
            g.id AS id,
            g.name AS name,
            g.affiliation AS aff,
            g.territory AS ter,
            collect(DISTINCT a.name) AS allies,
            collect(DISTINCT e.name) AS enemies,
            collect(DISTINCT c.name) AS cross
        """

        result = session.run(query)

        for record in result:
            allies = ", ".join(record["allies"]) if record["allies"] else "None"
            enemies = ", ".join(record["enemies"]) if record["enemies"] else "None"
            cross = ", ".join(record["cross"]) if record["cross"] else "None"

            text_to_embed = (
                f"Gang name: {record['name']}. "
                f"Affiliation: {record['aff']}. "
                f"Primary territory: {record['ter']}. "
                f"Allied gangs: {allies}. "
                f"Enemy gangs: {enemies}. "
                f"Cross-territory interactions with: {cross}."
            )

            response = client.embeddings.create(
                input=text_to_embed,
                model="text-embedding-3-small"
            )

            embedding = response.data[0].embedding

            session.run("""
                MATCH (g:Gang {id: $id})
                SET g.embedding = $embedding
            """, id=record["id"], embedding=embedding)

            print(f"✅ {record['id']} 임베딩 저장 완료")

        # -------------------------
        # 벡터 인덱스 생성
        # -------------------------
        print("🧱 벡터 인덱스 생성 중...")

        session.run("DROP INDEX gang_vector_index IF EXISTS")

        session.run("""
        CREATE VECTOR INDEX gang_vector_index
        FOR (g:Gang) ON (g.embedding)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
          }
        }
        """)

        print("🚀 montreal.gang_vector_index 생성 완료!")


if __name__ == "__main__":
    create_vector_embeddings()
    driver.close()