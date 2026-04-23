import os
import datetime
import networkx as nx
import matplotlib.pyplot as plt
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

# Connection settings
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not AUTH[1] or not OPENAI_API_KEY:
    raise ValueError("NEO4J_PASSWORD and OPENAI_API_KEY must be set in environment variables")

driver = GraphDatabase.driver(URI, auth=AUTH)
NEO4J_DB = "montreal"  # Specify the database name

# AI 구성 요소 초기화 (gpt-4o-mini는 데이터 스캔에 매우 효율적입니다)
llm = OpenAILLM(
    model_name="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    model_params={"temperature": 0}
)
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

graphrag = None
existing_gang_ids = [] 

def get_all_gang_ids():
    global existing_gang_ids
    with driver.session(database=NEO4J_DB) as session:
        result = session.run("MATCH (n:Gang) RETURN n.id AS id")
        existing_gang_ids = [record['id'] for record in result]

def initialize_retrievers():
    global graphrag
    try:
        # 1. Vector Retriever: 개별 조직의 속성 및 통계치 검색
        vector_retriever = VectorRetriever(
            driver=driver,
            index_name="gang_vector_index",
            embedder=embedder,
            return_properties=["id", "name", "affiliation", "territory", "ethnicity", 
                               "degreeCentrality", "betweennessCentrality", 
                               "closenessCentrality", "eigenvectorCentrality",  "clusteringCoeff"] 
        )

        # 2. VectorCypher Retriever: 관계 유형(Type)에 따른 전략적 맥락 추출
        # 관계의 종류(ALLIED, ENEMY 등)를 문장에 포함하여 AI가 상황을 인지하게 함
        vector_cypher_retriever = VectorCypherRetriever(
            driver=driver,
            index_name="gang_vector_index",
            retrieval_query="""
            MATCH (node)-[r]-(m:Gang)
            RETURN 
            "조직 " + node.name + " (지표 - 매개:" + toString(round(node.betweennessCentrality, 3)) + 
            ", 결속력:" + toString(round(node.clusteringCoeff, 3)) + ")는 " + 
            m.name + " (지표 - 매개:" + toString(round(m.betweennessCentrality, 3)) + 
            ", 실세:" + toString(round(m.eigenvectorCentrality, 3)) + ")와 " +
            type(r) + " 관계로 연결되어 있음." AS content,
            score
            """,
            embedder=embedder
        )

        # 3. Text2Cypher Retriever: 새로운 관계 체계를 반영한 예시 및 스키마
        # [업그레이드] ENEMY_OF, ALLIED_WITH, CROSS_TERRITORY를 활용한 고차원 쿼리 유도
        gang_examples = [
            "USER INPUT: 네트워크에서 내부 결속력이 가장 강해 일망타진하기 어려운 점조직은?\n"
            "CYPHER QUERY: MATCH (n:Gang) RETURN n.name, n.clusteringCoeff ORDER BY n.clusteringCoeff DESC LIMIT 3",
            
            "USER INPUT: BMF의 배후에서 실질적인 영향력을 행사하는 실세 조직은?\n"
            "CYPHER QUERY: MATCH (n:Gang {id:'BMF'})-[:ALLIED_WITH]-(m:Gang) RETURN m.name, m.eigenvectorCentrality ORDER BY m.eigenvectorCentrality DESC LIMIT 3",
            
            "USER INPUT: 진영 간 전쟁이 일어날 때 정보 흐름을 차단할 수 있는 가장 위험한 브로커는?\n"
            "CYPHER QUERY: MATCH (n:Gang) RETURN n.name, n.betweennessCentrality ORDER BY n.betweennessCentrality DESC LIMIT 1",
            
            "USER INPUT: 네트워크 전체에서 가장 빠르게 명령을 하달할 수 있는 조직은?\n"
            "CYPHER QUERY: MATCH (n:Gang) RETURN n.name, n.closenessCentrality ORDER BY n.closenessCentrality DESC LIMIT 3"
        ]

        text2cypher_retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            # [중요] 스키마에 새로 만든 관계 타입들을 명시적으로 교육
            neo4j_schema="""
            Node: Gang {
                id: STRING, name: STRING, affiliation: STRING, territory: STRING, 
                degreeCentrality: FLOAT (물리적 세력), 
                betweennessCentrality: FLOAT (교교량/브로커), 
                closenessCentrality: FLOAT (전파력), 
                eigenvectorCentrality: FLOAT (배후 실세 위상),
                clusteringCoeff: FLOAT (결속력)
            }, 
            Relationship: [PRIOR_CONTACT, ENEMY_OF, ALLIED_WITH, CROSS_TERRITORY]
            """,
            examples=gang_examples
        )

        # 4. ToolsRetriever: 도구의 역할을 관계 분석 중심으로 재정의
        tools_retriever = ToolsRetriever(
            driver=driver, 
            llm=llm,
            tools=[
                vector_retriever.convert_to_tool(
                    name="profile_and_metrics_lookup", 
                    description="조직의 기본 정보와 5대 지표(결속력, 실세도 등)를 직접 확인할 때 사용합니다."
                ),
                vector_cypher_retriever.convert_to_tool(
                    name="structural_context_analysis", 
                    description="주변 노드와의 관계 및 해당 클러스터의 밀집도를 파악할 때 사용합니다."
                ),
                text2cypher_retriever.convert_to_tool(
                    name="advanced_strategic_analytics", 
                    description="가장 위험한 브로커 탐색, 배후 세력 추적, 내부 결속력 기반 조직 복원력 분석 등에 사용합니다."
                )
            ]
        )

        # 5. Prompt 설정: 의미론적 관계 해석 지침 추가
        prompt_template = RagTemplate(
        template="""
        당신은 몬트리올 범죄 네트워크의 핵심을 꿰뚫어보는 수석 전략 분석관입니다.
        단순한 연결을 넘어, 관계의 **유형(Relationship Type)**이 갖는 의미를 심층 분석하세요.

        분석 가이드라인:
        1.  **지표 기반 분석 (0.0~1.0 정규화 기준)**: 
               - **betweenness**: 높을수록 지역/진영 간 유통을 담당하는 '핵심 브로커'입니다.
               - **eigenvector**: 높을수록 거대 조직(BMF 등)의 배후에 있는 '실세 조직'입니다.
               - **degree**: 높을수록 물리적 세력이 큰 '허브 조직'입니다.
               - **clusteringCoeff**: 높을수록 내부 결속력이 강한 '점조직'이며, 외부 침투가 어렵고 복원력이 뛰어납니다. 낮을수록 외부 확장에 치중한 개방적 구조입니다.
        2. **관계 영향 분석**: ENEMY_OF는 전쟁 중, ALLIED_WITH는 동맹, CROSS_TERRITORY 중 해당되는 관계를 확장의 증거로 해석하세요.
        3. **종합 추론**: 예를 들어, "A조직은 매개 중심성은 낮으나 결속력이 0.9로 매우 높아, 특정 지역을 요새화하고 있는 위험 조직임"과 같이 해석하세요.
        4. **증거 제시**: 반드시 수치적 근거를 포함하세요.

        Context: {context}
        Question: {query_text}`

        전략 분석 보고서:""",
        expected_inputs=["context", "query_text"]
        )

        graphrag = GraphRAG(llm=llm, retriever=tools_retriever, prompt_template=prompt_template)
        print("🚀 {NEO4J_DB} DB 기반 전략 분석 시스템 준비 완료")

    except Exception as e:
        print(f"❌ 초기화 실패: {e}")


def extract_gang_ids(text):
    """텍스트에서 (ID: BMF) 형태와 단순 ID를 모두 정밀 추출합니다."""
    # 1. (ID: BMF) 형태 추출
    ids_from_pattern = re.findall(r"\(ID:\s*([A-Za-z0-9_\-\.]+)\)", text)
    
    # 2. 리스트 형태 [ID1, ID2] 추출
    ids_from_list = re.findall(r"\[([A-Za-z0-9_\-\.\s,]+)\]", text)
    processed_list_ids = []
    for item in ids_from_list:
        processed_list_ids.extend([x.strip() for x in item.split(',')])

    # 3. 전체 텍스트에서 사전 정의된 ID와 대조
    all_potential = re.findall(r"([A-Za-z0-9_\-\.]+)", text.replace("_", " "))
    matched_ids = [word for word in all_potential if word in existing_gang_ids]

    final_ids = list(set(ids_from_pattern + processed_list_ids + matched_ids))
    return [fid for fid in final_ids if fid in existing_gang_ids]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def get_graph():
    nodes, edges = [], []
    with driver.session(database="montreal") as session:
        result = session.run("MATCH (n:Gang) OPTIONAL MATCH (n)-[r]->(m:Gang) RETURN n, r, m")
        for record in result:
            n_node = record['n']
            if n_node and not any(node['id'] == n_node['id'] for node in nodes):
                nodes.append({"id": n_node['id'], "label": "Gang", "properties": dict(n_node)})
            m_node = record['m']
            if m_node and not any(node['id'] == m_node['id'] for node in nodes):
                nodes.append({"id": m_node['id'], "label": "Gang", "properties": dict(m_node)})
            rel = record['r']
            if rel:
                edges.append({"source": n_node['id'], "target": m_node['id'], "relationship": rel.type})
    return jsonify({"nodes": nodes, "edges": edges})

@app.route("/debug/db")
def debug_db():
    with driver.session(database="montreal") as session:
        r = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t")
        return {"relations": [x["t"] for x in r]}

# --- [Flask API 엔드포인트] ---
@app.route('/query', methods=['POST'])
def query():
    user_question = request.json.get('question', '')
    if not graphrag:
        return jsonify({"answer": "시스템 초기화 중...", "used_nodes": []})

    try:
        # 성능 향상을 위해 gpt-4o-mini가 생성한 결과 활용
        result = graphrag.search(query_text=user_question, return_context=True)
        answer_text = result.answer
        
        # 리트리버가 가져온 로우 데이터(Context)도 ID 추출 대상에 포함하여 정확도 향상
        context_data = str(result.retriever_result)
        used_nodes = extract_gang_ids(answer_text + " " + context_data)
        
        return jsonify({
            "answer": answer_text,
            "used_nodes": used_nodes
        })
    except Exception as e:
        print(f"Query Error: {e}")
        return jsonify({"answer": "분석 오류 발생", "used_nodes": []})


 

if __name__ == '__main__':
    get_all_gang_ids()
    initialize_retrievers()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)