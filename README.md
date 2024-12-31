This is for my own learning and tinkering. My main goal is to avoid using frameworks such as `langchain` to build and better understand the structure of a rag application. Still, if you want to use this code, beyond python you need:


- [postgres](https://www.postgresql.org/)
- [pgvector extension](https://github.com/pgvector/pgvector)

On linux, if you get the error `postgres.h: No such file or directory` when trying to install pgvector, try running

```
sudo apt update
sudo apt install postgresql-server-dev-all
```
