import logging

from db_utils import pg_connection
from embed import embed_query


def retrieve_from_pgvector(query: str, db_name: str, table_name: str = 'pg_embeddings', retrieve_k = 5) -> list[str]:
    conn = pg_connection(db_name=db_name)
    embedded_query = embed_query(query).tolist()
    search_query = """
        SELECT chunk from {table_name}
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """.format(table_name=table_name)
    
    with conn.cursor() as cur:
        cur.execute(search_query, (embedded_query, retrieve_k))
        results = cur.fetchall()

    conn.close()

    return [doc[0] for doc in results]