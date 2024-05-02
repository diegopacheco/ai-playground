### Build 
```bash
./mvnw clean install 
```
```
docker run --rm --name pgvector-container -e POSTGRES_PASSWORD=password \
 -p 5432:5432 -v pgvector-data:/var/lib/postgresql/data \
 -v $(pwd)/src/main/resources/:/docker-entrypoint-initdb.d/ pgvector/pgvector:pg16
```
pgvector16 seems to stop working:
https://github.com/docker-library/postgres/issues/1099#issuecomment-1593228770
https://stackoverflow.com/questions/76558498/initdb-error-program-postgres-is-needed-by-initdb-but-was-not-found-using-ci

```
❯ docker run --rm --name pgvector-container -e POSTGRES_PASSWORD=password \
 -p 5432:5432 -v pgvector-data:/var/lib/postgresql/data \
 pgvector/pgvector:pg15
popen failure: Cannot allocate memory
initdb: error: program "postgres" is needed by initdb but was not found in the same directory as "/usr/lib/postgresql/15/bin/initdb"
```
```
❯ docker run --rm --name pgvector-container -e POSTGRES_PASSWORD=password \
 -p 5432:5432 -v pgvector-data:/var/lib/postgresql/data \
 pgvector/pgvector:pg16
popen failure: Cannot allocate memory
initdb: error: program "postgres" is needed by initdb but was not found in the same directory as "/usr/lib/postgresql/16/bin/initdb"
```