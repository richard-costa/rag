import logging

from db_utils import pg_connection
from embed import embed_query
from psycopg import sql
from dataclasses import dataclass

@dataclass
class Distances:
    cosine = '<=>'
    l2 = '<->'

def retrieve_from_pgvector(query: str, 
                           db_name: str,
                           table_name: str = 'pg_embeddings',
                           with_score: bool = True, 
                           distance_operator = Distances.cosine,
                           retrieve_k = 5) -> list[tuple[str, None]] | list[tuple[str, float]]:
    
    conn = pg_connection(db_name=db_name)
    embedded_query = embed_query(query).tolist()
    
    search_query = sql.SQL("""
        SELECT chunk from {table_name}
        ORDER BY embedding {distance_operator} %s::vector
        LIMIT %s
        """).format(table_name=sql.Identifier(table_name), 
                   distance_operator=sql.SQL(distance_operator))

    if with_score:
        search_query = sql.SQL("""
                        SELECT 
                            chunk, 
                            embedding {distance_operator} %s::vector AS score
                        FROM 
                            {table_name}
                        ORDER BY 
                            score
                        LIMIT 
                            %s;
                        """).format(table_name=sql.Identifier(table_name), 
                                   distance_operator=sql.SQL(distance_operator))
        

        
    with conn.cursor() as cur:
        cur.execute(search_query, (embedded_query, retrieve_k))
        results = cur.fetchall()

    conn.close()

    return results 

if __name__ == '__main__':
    res = retrieve_from_pgvector('physics', 'test_db', with_score=True)
    print(res)
    print(type(res))
    print(Distances.cosine)