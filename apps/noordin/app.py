import os
import re
import json
from flask import Flask, render_template, jsonify, request
from neo4j import GraphDatabase
from dotenv import load_dotenv

# GraphRAG 컴포넌트
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever, Text2CypherRetriever, ToolsRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

load_dotenv()

app = Flask(__name__)

# 1. Connection and DB Settings
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PASSWORD or not OPENAI_API_KEY:
    raise ValueError("NEO4J_PASSWORD and OPENAI_API_KEY must be set in environment variables")

AUTH = ("neo4j", PASSWORD)
NEO4J_DB = "noordin"  # Noordin instance

driver = GraphDatabase.driver(URI, auth=AUTH)

# 2. AI 구성 요소 초기화
llm = OpenAILLM(
    model_name="gpt-4o-mini", 
    api_key=OPENAI_API_KEY, 
    model_params={"temperature": 0}
)
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

graphrag = None
existing_person_ids = []

# DB 내 모든 Person ID 로드
def get_all_person_ids():
    global existing_person_ids
    with driver.session(database=NEO4J_DB) as session:
        result = session.run("MATCH (p:Person) RETURN p.id AS id")
        existing_person_ids = [r["id"] for r in result]

def initialize_retrievers():
    global graphrag
    try:
        # 1. Vector Retriever: 속성은 노드 속성명이므로 그대로 유지 (보통 소문자)
        vector_retriever = VectorRetriever(
            driver=driver,
            index_name="person_vector_index",
            embedder=embedder,
            return_properties=["id", "name", "group", "status", "role"] 
        )

        # 2. VectorCypher Retriever: CASE 문 내부의 관계명을 대문자로 수정
        vector_cypher_retriever = VectorCypherRetriever(
            driver=driver,
            index_name="person_vector_index",
            retrieval_query="""
            MATCH (node)-[r]-(m:Person)
            WITH node, m, type(r) AS rel,
                CASE type(r)
                    WHEN 'KINSHIP' THEN 1.0
                    WHEN 'SOULMATE' THEN 0.95
                    WHEN 'BUSINESS' THEN 0.85
                    WHEN 'CO_TRAINED' THEN 0.8
                    WHEN 'CO_OPERATION' THEN 0.75
                    WHEN 'COMMUNICATE' THEN 0.7
                    WHEN 'FRIENDSHIP' THEN 0.65
                    WHEN 'CLASSMATES' THEN 0.55
                    WHEN 'CO_MEETING' THEN 0.4
                    ELSE 0.25
                END AS importance
            RETURN
            node.name + " (ID: " + node.id + ")는 " +
            m.name + " (ID: " + m.id + ")와 '" + rel +
            "' 관계를 맺고 있으며, 이 관계의 전략적 중요도는 " +
            toString(importance) + "이다." AS content,
            importance AS score
            """,
            embedder=embedder
        )

        # 3. Text2Cypher Retriever: 관계 타입 리스트와 예시 쿼리 수정
        # [수정] 대문자 및 언더바(_) 적용
        rel_types = "CO_OPERATION, COMMUNICATE, CLASSMATES, CO_TRAINED, FRIENDSHIP, CO_MEETING, CO_LOCATION, KINSHIP, BUSINESS, SOULMATE"
        
        person_examples = [
            """USER INPUT: 단순히 아는 사람이 많은 것이 아니라, KINSHIP이나 SOULMATE처럼 강력한 신뢰로 연결된 핵심 실세 5명은?
            CYPHER QUERY:
            MATCH (n:Person)-[r]-()
            WITH n,
                sum(CASE type(r)
                    WHEN 'KINSHIP' THEN 10
                    WHEN 'SOULMATE' THEN 8
                    WHEN 'BUSINESS' THEN 5
                    WHEN 'FRIENDSHIP' THEN 3
                    ELSE 1
                END) AS trust_score
            RETURN n.id, n.name, trust_score
            ORDER BY trust_score DESC
            LIMIT 5
            """,

            """USER INPUT: 함께 훈련(CO_TRAINED)받고 실제 작전(CO_OPERATION)까지 수행하여 팀워크가 검증된 인물들의 조합을 찾아줘.
            CYPHER QUERY:
            MATCH (n:Person)-[:CO_TRAINED]-(m:Person)
            MATCH (n)-[:CO_OPERATION]-(m)
            RETURN n.id, m.id, '검증된 작전팀' AS status
            """,

            """USER INPUT: Noordin Top과 직접 연결되지는 않았지만, 통신(COMMUNICATE) 관계를 통해 그에게 정보를 전달할 수 있는 중간 연락책은?
            CYPHER QUERY:
            MATCH (n:Person {id: 'Noordin_Top'})-[:COMMUNICATE]-(broker)-[:COMMUNICATE]-(target)
            WHERE NOT (n)-[:COMMUNICATE]-(target) AND target <> n
            RETURN DISTINCT broker.id, broker.name
            """,

            """USER INPUT: 같은 장소(CO_LOCATION)에 머물렀거나 회의(CO_MEETING)에 참석하여 물리적 접점이 확인된 그룹은?
            CYPHER QUERY:
            MATCH (n:Person)-[r:CO_LOCATION|CO_MEETING]-(m:Person)
            RETURN n.id, type(r), m.id
            """,

            """USER INPUT: Noordin Top과 3가지 이상의 서로 다른 관계 유형으로 복잡하게 얽혀 있어, 관계를 끊기 어려운 핵심 측근은?
            CYPHER QUERY:
            MATCH (n:Person {id: 'Noordin_Top'})-[r]-(m:Person)
            WITH m, count(DISTINCT type(r)) AS rel_count
            WHERE rel_count >= 3
            RETURN m.id, m.name, rel_count
            ORDER BY rel_count DESC
            """
        ]

        text2cypher_retriever = Text2CypherRetriever(
            driver=driver, 
            llm=llm,
            neo4j_schema=f"Node: Person {{id: STRING, name: STRING, group: STRING}}, Relationship: [{rel_types}]",
            examples=person_examples
        )

        # 4. ToolsRetriever (설명 부분에 대문자 반영)
        tools_retriever = ToolsRetriever(
            driver=driver, llm=llm,
            tools=[
                vector_retriever.convert_to_tool(
                    name="person_info", 
                    description="특정 테러리스트의 이름, 소속 그룹, 역할 등 개별 프로필 정보를 찾을 때 사용합니다."
                ),
                vector_cypher_retriever.convert_to_tool(
                    name="connection_finder", 
                    description="인물 간의 구체적인 관계 유형(KINSHIP, COMMUNICATE, CO_TRAINED 등)과 주변 네트워크를 파악할 때 사용합니다."
                ),
                text2cypher_retriever.convert_to_tool(
                    name="network_analysis", 
                    description="네트워크 전체의 통계, 핵심 허브 분석, 특정 관계 유형의 빈도 측정 등에 사용합니다."
                )
            ]
        )

        # 5. Prompt 설정: 가이드라인 내 관계명 수정
        prompt_template = RagTemplate(
        template="""
        당신은 'Noordin Top 테러 네트워크' 구조 분석 전문가입니다.
        단순한 데이터 나열이 아닌, 관계의 '강도'와 '의미'를 중심으로 보고서를 작성하세요.

        분석 프레임워크:
        1. **신뢰 구조(Trust Backbone)**: `KINSHIP`, `SOULMATE`, `BUSINESS` 관계를 가진 인물은 체포 시에도 배신 가능성이 낮은 핵심 계보로 분류합니다.
        2. **운영 구조(Operational Line)**: `CO_TRAINED`, `CO_OPERATION` 관계는 실제 테러 수행 능력을 의미합니다.
        3. **통신 및 사회 구조(Social/Comm Line)**: `COMMUNICATE`, `CLASSMATES`, `FRIENDSHIP`은 정보의 흐름과 신규 조직원 포섭 경로를 나타냅니다.
        4. **취약점 분석**: 특정 인물(Hub)이 제거되었을 때 네트워크가 얼마나 파편화될지(Fragmentation) 예측하십시오.

        답변 형식:
        - 반드시 "이름 (ID: id)" 형식을 사용하세요.
        - 수치 데이터(점수, 연결 수 등)를 근거로 제시하세요.

        Context: {context}
        Question: {query_text}

        전문 분석 보고서:
        """,
        expected_inputs=["context", "query_text"]
        )

        graphrag = GraphRAG(llm=llm, retriever=tools_retriever, prompt_template=prompt_template)
        print(f"✅ 'noordin' DB 기반 GraphRAG 로드 완료 (대문자 관계 유형 반영)")

    except Exception as e:
        print(f"❌ 초기화 실패: {e}")

def extract_person_ids(text):
    """텍스트에서 실제 존재하는 인물 ID를 정밀 추출합니다."""
    # (ID: ID값) 패턴 추출
    ids_from_pattern = re.findall(r"\(ID:\s*([A-Za-z0-9_\-\.]+)\)", text)
    # 일반 텍스트 내 매칭
    all_potential = re.findall(r"([A-Za-z0-9_\-\.]+)", text.replace("_", " "))
    matched_ids = [word for word in all_potential if word in existing_person_ids]
    
    return list(set(ids_from_pattern + matched_ids))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def get_graph():
    """Noordin 네트워크 전체 시각화 데이터 반환"""
    nodes, edges = [], []
    with driver.session(database=NEO4J_DB) as session:
        # 모든 Person 노드와 그들 사이의 관계를 가져옴
        result = session.run("MATCH (n:Person) OPTIONAL MATCH (n)-[r]->(m:Person) RETURN n, r, m")
        for record in result:
            n_node = record['n']
            if n_node and not any(node['id'] == n_node['id'] for node in nodes):
                nodes.append({"id": n_node['id'], "label": "Person", "properties": dict(n_node)})
            
            m_node = record['m']
            if m_node and not any(node['id'] == m_node['id'] for node in nodes):
                nodes.append({"id": m_node['id'], "label": "Person", "properties": dict(m_node)})
            
            rel = record['r']
            if rel:
                edges.append({
                    "id": rel.element_id,
                    "source": n_node['id'], 
                    "target": m_node['id'], 
                    "relationship": rel.type
                })
    return jsonify({"nodes": nodes, "edges": edges})

@app.route('/query', methods=['POST'])
def query():
    user_question = request.json.get('question', '')
    if not graphrag:
        return jsonify({"answer": "시스템 초기화 오류", "used_nodes": []})

    try:
        # GraphRAG 검색
        result = graphrag.search(query_text=user_question, return_context=True)
        answer_text = result.answer
        context_data = str(result.retriever_result)
        
        # ID 추출 (답변 + 컨텍스트)
        used_nodes = extract_person_ids(answer_text + " " + context_data)
        
        return jsonify({
            "answer": answer_text,
            "used_nodes": used_nodes
        })
    except Exception as e:
        print(f"Query Error: {e}")
        return jsonify({"answer": f"분석 오류: {str(e)}", "used_nodes": []})
    

if __name__ == '__main__':
    get_all_person_ids()
    initialize_retrievers()
    app.run(host="0.0.0.0", port=8000, debug=True) # 포트 8000 사용