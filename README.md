This is for my own learning and tinkering. My main goal is to avoid using frameworks such as `langchain` to build and better understand the structure of a rag application. Still, if you want to use this code, beyond python you need:


- [postgres](https://www.postgresql.org/)
- [pgvector extension](https://github.com/pgvector/pgvector)

On linux, if you get the error `postgres.h: No such file or directory` when trying to install pgvector, try running

```
sudo apt update
sudo apt install postgresql-server-dev-all
```

# `.env file`

The `.env`file should have the following entries

```
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = '5432'
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = <your password>
```

# Installing from `pyproject.toml`

I'm not sure whether to keep the `requirements.txt` file or not, so if there's only a `pyproject.toml` file in this repo, a fast and easy way to install the project's dependencies is to use [uv](https://github.com/astral-sh):

```
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -r pyproject.toml
```

# Ollama

The generation in RAG is currently being done using [ollama](https://ollama.com/) and its python library [ollama-python](https://github.com/ollama/ollama-python). See ollama installation instructions [here](https://github.com/ollama/ollama?tab=readme-ov-file).