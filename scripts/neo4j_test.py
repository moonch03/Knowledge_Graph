import os
import time
import requests
from typing import List
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.chains import RetrievalQA
# Note: Assuming custom or specific library for Clova
# from clova_library import ClovaXEmbeddings, ChatClovaX 

# --- PDF Processing and Segmentation ---

def read_pdf_text(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    texts = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(texts).strip()

def chunk_by_limit(text: str, limit: int) -> List[str]:
    """Chunks text by character limit."""
    parts = text.split("\n\n")
    batches, cur = [], ""
    for p in parts:
        add = p if not cur else ("\n\n" + p)
        if len(cur) + len(add) <= limit:
            cur += add
        else:
            if cur: batches.append(cur)
            cur = p if len(p) <= limit else ""
            if len(p) > limit:
                print(f"[WARN] A single paragraph of {len(p)} chars is longer than the limit {limit}, it will be chunked bluntly.")
                for i in range(0, len(p), limit):
                    batches.append(p[i:i+limit])
    if cur: batches.append(cur)
    return batches

def clova_segment(text: str, endpoint: str, headers: dict, **kwargs) -> List[str]:
    """Segments text using CLOVA Studio Segmentation API."""
    payload = {
        "text": text,
        "alpha": kwargs.get("alpha", -100),
        "segCnt": kwargs.get("segCnt", -1),
        "postProcess": kwargs.get("postProcess", True),
        "postProcessMaxSize": kwargs.get("postProcessMaxSize", 1000),
        "postProcessMinSize": kwargs.get("postProcessMinSize", 200),
    }
    retry, backoff_sec = 3, 1.5
    for attempt in range(1, retry + 1):
        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            data = r.json()
            if r.status_code == 200 and data.get("status", {}).get("code") == "20000":
                return ["\n".join(sents).strip() for sents in data["result"].get("topicSeg", []) if sents]
            raise RuntimeError(f"Segmentation API error: HTTP {r.status_code}, body={data}")
        except Exception as e:
            if attempt == retry: raise e
            time.sleep(backoff_sec * attempt)
    return []

# --- Knowledge Graph Extraction ---

class Node(BaseModel):
    id: str = Field(description="Name or ID of the node (entity)")
    type: str = Field(description="Type of the entity (e.g., Person, Organization, Concept)")

class Relationship(BaseModel):
    source: Node = Field(description="Source node of the relationship")
    target: Node = Field(description="Target node of the relationship")
    type: str = Field(description="Type of the relationship (e.g., IS_A, WORKS_FOR, CONTAINS)")

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="List of all nodes in the graph")
    relationships: List[Relationship] = Field(description="List of all relationships in the graph")

def get_graph_extraction_chain(llm):
    parser = JsonOutputParser(pydantic_object=KnowledgeGraph)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Identify key entities (nodes) and relationships (edges) from the given text.
        Output all extracted nodes and relationships in JSON format.
        Use common terms for node types (e.g., 'Person', 'Organization').
        Use uppercase snake_case for relationship types (e.g., 'WORKS_FOR').
        {format_instructions}
        """),
        ("human", "Extract a knowledge graph from the following text:\n\n{input}")
    ])
    return prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

# --- Ingestion ---

def ingest_chunks_to_neo4j(chunks: List[str], graph: Neo4jGraph, graph_extraction_chain, embedder):
    """Ingests text chunks and extracted graph data into Neo4j."""
    graph.query("MATCH (n) DETACH DELETE n")
    print("Cleared existing data from Neo4j.")

    for i, chunk_text in enumerate(chunks):
        if not chunk_text or chunk_text.isspace(): continue
        
        print(f"--- Processing chunk {i+1}/{len(chunks)} ---")
        chunk_embedding = embedder.embed_query(chunk_text)
        graph.query(
            "MERGE (c:Chunk {id: $chunk_id}) SET c.text = $text, c.embedding = $embedding",
            params={"chunk_id": f"chunk_{i}", "text": chunk_text, "embedding": chunk_embedding}
        )

        try:
            extracted_data = graph_extraction_chain.invoke({"input": chunk_text})
            for node in extracted_data.get("nodes", []):
                graph.query(f"MERGE (n:`{node['type']}` {{id: $id}})", params={"id": node['id']})
                graph.query(
                    f"MATCH (c:Chunk {{id: $chunk_id}}), (n:`{node['type']}` {{id: $node_id}}) MERGE (c)-[:CONTAINS]->(n)",
                    params={"chunk_id": f"chunk_{i}", "node_id": node['id']}
                )
            for rel in extracted_data.get("relationships", []):
                graph.query(
                    f"MATCH (s:`{rel['source']['type']}` {{id: $sid}}), (t:`{rel['target']['type']}` {{id: $tid}}) MERGE (s)-[:`{rel['type']}`]->(t)",
                    params={"sid": rel['source']['id'], "tid": rel['target']['id']}
                )
        except Exception as e:
            print(f"Error extracting graph from chunk {i}: {e}")

if __name__ == "__main__":
    # Example usage or test logic could be added here
    pass
