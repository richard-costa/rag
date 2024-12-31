import logging

from db_utils import get_pg_connection
from embed import embed_query


def retrieve_from_pgvector(query: str, db_name: str, table_name: str = 'pg_embeddings', retrieve_k = 5):
    conn = get_pg_connection(db_name=db_name)
    cur = conn.cursor()
    embedded_query = embed_query(query).tolist()
    search_query = """
        SELECT chunk from {table_name}
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """.format(table_name=table_name)
    try:
        cur.execute(search_query, (embedded_query, retrieve_k))
        results = cur.fetchall()
        return [doc[0] for doc in results]
    except Exception as e:
        logging.error(e)
        print(f"{e.__class__.__name__}: {e}")
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    query = 'copenhagen'
    print(retrieve_from_pgvector(query, 'test_db'))