import logging
import os

import psycopg2
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)

def get_pg_connection(db_name: str):
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        port=int(os.getenv('POSTGRES_PORT')),
        database=db_name,
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD')
    )


def create_db(db_name: str):    
    conn = get_pg_connection(db_name=os.getenv('POSTGRES_USER'))
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute(f"CREATE DATABASE {db_name}")
        logging.info(f"Database {db_name} created successfully")
    except psycopg2.errors.DuplicateDatabase:
        logging.warning(f"Database '{db_name}' already exists.")
    except Exception as e:
        logging.error(f"Error creating database '{db_name}': {e}")
    finally:
        cur.close()
        conn.close()

def create_pgvector_extension(db_name: str) -> None:
    conn = get_pg_connection(db_name=db_name)
    cur = conn.cursor()
    conn.autocommit = True
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logging.info("pgvector extension created")

    except Exception as e:
        logging.error(e)
    finally:
        cur.close()
        conn.close()

def delete_db(db_name: str):
    conn = get_pg_connection(db_name=os.getenv('POSTGRES_USER'))
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute(f"DROP DATABASE {db_name}")
        logging.info(f"Database {db_name} deleted")
    
    except psycopg2.errors.InvalidCatalogName:
        logging.warning(f"Database '{db_name}' does not exist.")
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
    finally:
        cur.close()
        conn.close()

def list_databases() -> list[str]:
    conn = get_pg_connection(db_name=os.getenv('POSTGRES_USER'))
    cur = conn.cursor()
    try:
        cur.execute("SELECT datname FROM pg_database")
        databases = cur.fetchall()
        return [d[0] for d in databases]
    except Exception as e:
        logging.error(e)
        print(f"{e.__class__.__name__}: {e}")
    finally:
        cur.close()
        conn.close()


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

def create_embeddings_table(db_name: str, chunks: list[str], embeddings: list[list[float]], table_name: str = 'pg_embeddings'):
    conn = get_pg_connection(db_name=db_name)
    cur = conn.cursor()
    conn.autocommit = True

    create_query = '''
            CREATE TABLE {table_name} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(), 
                chunk text,
                embedding vector
            );
            '''.format(table_name=table_name)

    try:
        cur.execute(create_query)
        insert_query = 'INSERT INTO {table_name} (chunk, embedding) VALUES (%s, %s::vector);'.format(table_name=table_name)

        for chunk, embedding in zip(chunks, embeddings):
            cur.execute(insert_query, (chunk, embedding,))
        
        logging.info(f"Embeddings table '{table_name}' created and data inserted.")
    except Exception as e:
        logging.error(e)

    finally:
        cur.close()
        conn.close()

