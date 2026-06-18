import os
from graph_rag import GraphRAGManager
import config

def main():
    print('Initializing Graph RAG Manager...')
    graph_rag = GraphRAGManager(
        enabled=config.ENABLE_GRAPH_RAG,
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
        max_hops=config.GRAPH_MAX_HOPS,
        max_query_entities=config.GRAPH_MAX_QUERY_ENTITIES,
        max_graph_chunks=config.GRAPH_TOP_K,
        max_chunks_per_entity=config.GRAPH_MAX_CHUNKS_PER_ENTITY,
        semantic_merge_enabled=config.GRAPH_SEMANTIC_MERGE_ENABLED,
        semantic_merge_threshold=config.GRAPH_SEMANTIC_MERGE_THRESHOLD,
        extraction_timeout=config.GRAPH_EXTRACTION_TIMEOUT,
        extraction_retries=config.GRAPH_EXTRACTION_RETRIES,
    )
    
    if not graph_rag.available:
        print('Graph RAG is not available. Check Neo4j connection.')
        return
        
    print('Exporting Knowledge Graph...')
    md_content = graph_rag.export_to_markdown()
    
    output_path = 'knowledge_graph_export.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f'Graph exported successfully to: {os.path.abspath(output_path)}')

if __name__ == '__main__':
    main()
