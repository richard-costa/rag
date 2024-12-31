from retrieval import retrieve_from_pgvector

if __name__ == '__main__':
    query = 'copenhagen'
    print(retrieve_from_pgvector(query, 'test_db'))