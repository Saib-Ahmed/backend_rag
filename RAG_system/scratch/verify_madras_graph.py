import sys
import os
from pathlib import Path

# Add the parent directory (RAG_system) to the python path
sys.path.append(str(Path(__file__).parent.parent))

import config
from neo4j import GraphDatabase

def main():
    uri = config.NEO4J_URI
    user = config.NEO4J_USER
    password = config.NEO4J_PASSWORD
    database = config.NEO4J_DATABASE

    print(f"Connecting to Neo4j at {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return

    doc_name = "HC_Madras_2023_Feedback Infra Pvt ltd.pdf"
    
    with driver.session(database=database) as session:
        # 1. Fetch chunks
        print("\n--- Chunks ---")
        chunks = list(session.run(
            "MATCH (c:Chunk) WHERE c.source = $source RETURN c.chunk_id AS chunk_id, c.page_label AS page, c.section AS section, size(c.text) AS text_len",
            source=doc_name
        ))
        print(f"Found {len(chunks)} chunks.")
        for ck in chunks[:10]:
            print(dict(ck))
        if len(chunks) > 10:
            print("...")

        # 2. Fetch entities mentioned by these chunks
        print("\n--- Entities Mentioned ---")
        entities = list(session.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.source = $source
            RETURN DISTINCT e.name AS name, e.type AS type, e.aliases AS aliases, e.entity_key AS entity_key
            ORDER BY e.type, e.name
            """,
            source=doc_name
        ))
        print(f"Found {len(entities)} entities.")
        for entity in entities:
            print(f"- **{entity['name']}** ({entity['type']}) | Key: {entity['entity_key']} | Aliases: {entity['aliases']}")

        # 3. Fetch relationships
        print("\n--- Relationships ---")
        relationships = list(session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE $source IN r.sources
            RETURN a.name AS source_name, type(r) AS relation, b.name AS target_name, r.confidence AS confidence, r.evidence AS evidence
            """,
            source=doc_name
        ))
        print(f"Found {len(relationships)} relationships.")
        for rel in relationships:
            print(f"- {rel['source_name']} -> **{rel['relation']}** -> {rel['target_name']} (Conf: {rel['confidence']})")
            print(f"  Evidence: {rel['evidence']}")

    driver.close()

if __name__ == "__main__":
    main()
