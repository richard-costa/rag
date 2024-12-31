from db_utils import (create_db, create_embeddings_table,
                      create_pgvector_extension, delete_all_databases,
                      insert_data_into_table)

if __name__ == '__main__':
    delete_all_databases()
    db_name = 'test_db'

    create_db(db_name=db_name)
    create_pgvector_extension(db_name=db_name)
    create_embeddings_table(db_name)

    # text_chunks = ['hello', 'world', 'how', 'are', 'you']
    # text_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
    # insert_data_into_table(db_name, text_chunks, text_embeddings)