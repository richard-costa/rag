import logging
import os

from dotenv import load_dotenv
from psycopg import connect, sql
from psycopg.connection import Connection as PGConnection
from psycopg.errors import (DuplicateDatabase, DuplicateTable,
                            InvalidCatalogName)

logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)

def pg_connection(db_name: str) -> PGConnection:
    return connect(
        host=os.getenv('POSTGRES_HOST'),
        port=int(os.getenv('POSTGRES_PORT')),
        dbname=db_name,
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD')
    )

        
def create_db(db_name: str):
    query = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
    try:
        conn = pg_connection(db_name=os.getenv('POSTGRES_USER'))
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute(query)

        logging.info(f"Database '{db_name}' created successfully")
    except DuplicateDatabase:
        logging.warning(f"Database '{db_name}' already exists.")

    finally:
        conn.close()


def create_pgvector_extension(db_name: str):
    conn = pg_connection(db_name=db_name)
    cur = conn.cursor()
    conn.autocommit = True
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    logging.info("pgvector extension created")
    cur.close()
    conn.close()

def delete_db(db_name: str):
    conn = pg_connection(db_name=os.getenv('POSTGRES_USER'))
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute(f"DROP DATABASE {db_name}")
        logging.info(f"Database {db_name} deleted")
    
    except InvalidCatalogName:
        logging.warning(f"Database '{db_name}' does not exist.")

    finally:
        cur.close()
        conn.close()

def list_databases() -> list[str]:
    conn = pg_connection(db_name=os.getenv('POSTGRES_USER'))
    cur = conn.cursor()

    cur.execute("SELECT datname FROM pg_database")
    databases = cur.fetchall()
    cur.close()
    conn.close()

    return [d[0] for d in databases]


def delete_all_databases():
    databases = list_databases()
    to_remove = ['postgres', 'template0', 'template1']
    if os.getenv('POSTGRES_USER') not in databases:
        to_remove.append(os.getenv('POSTGRES_USER'))
    for db in to_remove:
        databases.remove(db)
    for db in databases:
        delete_db(db)
    
    logging.info("Deleted all databases")

def create_embeddings_table(db_name: str, table_name: str = 'pg_embeddings'):
    conn = pg_connection(db_name=db_name)
    cur = conn.cursor()
    conn.autocommit = True
    create_query = '''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(), 
                chunk text,
                embedding vector
            );
            '''.format(table_name=table_name)
    
    cur.execute(create_query)
    logging.info(f"Embeddings table '{table_name}' created.")

    cur.close()
    conn.close()

def insert_data_into_table(db_name: str, chunks: list[str], embeddings: list[list[float]], table_name: str = 'pg_embeddings'):
    conn = pg_connection(db_name=db_name)
    cur = conn.cursor()
    conn.autocommit = True
    insert_query = 'INSERT INTO {table_name} (chunk, embedding) VALUES (%s, %s::vector);'.format(table_name=table_name)
    
    failed_chunks = 0

    for chunk, embedding in zip(chunks, embeddings):
        try:
            cur.execute(insert_query, (chunk, embedding,))
        except Exception as e:
            logging.error(f"Error inserting data into table '{table_name}': {e}")
            failed_chunks += 1

    logging.info(f"Data inserted into table '{table_name}'. Failed chunks: {failed_chunks}")

    cur.close()
    conn.close()
