from dataclasses import dataclass

from psycopg import sql

from db_utils import pg_connection
from embed import embed_query


@dataclass
class Distances:
    cosine = '<=>'
    l2 = '<->'


def semantic(
        query: str, 
        db_name: str,
        table_name: str = 'pg_embeddings_test',
        with_score: bool = True, 
        distance_operator = Distances.cosine,
        retrieve_k = 5) -> list[tuple[str, None]] | list[tuple[str, float]]:

    conn = pg_connection(db_name=db_name)
    embedded_query = embed_query(query)
    
    search_stmt = sql.SQL("""\
        SELECT chunk from {table_name}
        ORDER BY embedding {distance_operator} %s::vector
        LIMIT %s
        """).format(table_name=sql.Identifier(table_name), 
                    distance_operator=sql.SQL(distance_operator))

    if with_score:
        search_stmt = sql.SQL("""\
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
        cur.execute(search_stmt, (embedded_query, retrieve_k))
        results = cur.fetchall()

    conn.close()

    return results 

def hybrid(
        query: str,
        db_name: str,
        lang: str = 'english',
        rrf_k = 60,
        retrieve_k = 5,
        table_name: str = 'pg_embeddings_test',
        distance_operator = Distances.cosine,
        with_score: bool = True) -> list[tuple[str, None]] | list[tuple[str, float]]:
    
    conn = pg_connection(db_name=db_name)
    embedded_query = embed_query(query)

    search_stmt = sql.SQL("""\
        SELECT
            searches.id,
            searches.chunk,
            SUM(1.0  / (searches.rank + %s)) AS score -- Inlined rrf_score logic
        FROM (
            (
                SELECT
                    id,
                    chunk, 
                    RANK() OVER (ORDER BY embedding {distance_operator} %s::vector) AS rank
                FROM {table_name}
                ORDER BY embedding {distance_operator} %s::vector
                LIMIT 20
            )
            UNION
            (
                SELECT
                    id,
                    chunk, 
                    RANK() OVER (ORDER BY ts_rank_cd(to_tsvector(chunk), plainto_tsquery(%s)) DESC) AS rank
                FROM {table_name}
                WHERE to_tsvector(%s, chunk) @@ plainto_tsquery(%s, %s)
                ORDER BY rank
                LIMIT 20
            )
        ) searches
        GROUP BY searches.id, searches.chunk
        ORDER BY score DESC
        LIMIT %s;
        """).format(table_name=sql.Identifier(table_name), 
                    distance_operator=sql.SQL(distance_operator))
        
    with conn.cursor() as cur:
        params = (
            rrf_k,
            embedded_query,
            embedded_query,
            query,
            lang,
            lang,
            query,
            retrieve_k
        )

        cur.execute(search_stmt, params)
        results = cur.fetchall()

    conn.close()

    return results


if __name__ == "__main__":
    res = hybrid('copenhagen', 'test_db')
    for r in res:
        print(f"""\
              chunk id: {r[0]}
              chunk: {r[1]}
              score: {r[2]}
              """)