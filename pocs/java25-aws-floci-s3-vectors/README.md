# Java 25 with Floci S3 Vectors

A compact REST service that stores and searches vectors through the AWS SDK for Java and a local [Floci](https://floci.io/aws/) AWS endpoint.

## Stack

| Component | Version |
|---|---:|
| Java | 25 |
| Spring Boot | 4.1.0 |
| Maven | 3.9.12 |
| AWS SDK for Java | 2.49.3 |
| Springdoc OpenAPI | 3.0.3 |
| Floci | latest container image |
| Podman | 5+ |

## Architecture

```text
REST client
    |
    v
Spring Boot :8080
    |
    v
AWS SDK for Java
    |
    v
Floci :4566
    |
    v
S3 vector bucket / index
```

The service creates the `product-vectors` bucket and `products` cosine index when they are first needed. The index stores three-dimensional `float32` vectors.

## Requirements

- Java 25 and Maven for local builds
- Podman
- podman-compose
- curl

## Start

```bash
./start.sh
```

The script builds both Maven stages, starts Floci and the application, and waits for the S3 Vectors infrastructure to become available.

Host ports can be changed when another service already uses the defaults:

```bash
APP_PORT=8090 FLOCI_PORT=4570 ./start.sh
APP_PORT=8090 ./test.sh
```

| Address | Purpose |
|---|---|
| http://localhost:8082/swagger-ui.html | Swagger UI |
| http://localhost:8082/v3/api-docs | OpenAPI document |
| http://localhost:8082/api/v1/status | Service status |
| http://localhost:4568 | Floci AWS endpoint |

## Verify the REST API

```bash
./test.sh
```

The script stores three vectors, retrieves them, performs cosine nearest-neighbor search, checks the responses, and removes the vectors.

## REST calls

Store or replace a vector:

```bash
curl -X PUT http://localhost:8082/api/v1/vectors/java \
  -H "Content-Type: application/json" \
  -d '{"values":[1.0,0.0,0.0]}'
```

Get a vector:

```bash
curl http://localhost:8082/api/v1/vectors/java
```

Find the two nearest vectors:

```bash
curl -X POST http://localhost:8082/api/v1/vectors/search \
  -H "Content-Type: application/json" \
  -d '{"values":[0.9,0.1,0.0],"topK":2}'
```

Delete a vector:

```bash
curl -X DELETE http://localhost:8082/api/v1/vectors/java
```

## Configuration

| Environment variable | Default |
|---|---|
| `APP_PORT` | `8082` |
| `FLOCI_PORT` | `4568` |
| `S3_VECTORS_ENDPOINT` | `http://localhost:4566` |
| `AWS_REGION` | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | `test` |
| `AWS_SECRET_ACCESS_KEY` | `test` |
| `S3_VECTORS_BUCKET` | `product-vectors` |
| `S3_VECTORS_INDEX` | `products` |
| `S3_VECTORS_DIMENSION` | `3` |

## Build and test

```bash
mvn clean test
mvn clean package
```

## Stop

```bash
./stop.sh
```

The named `floci-data` volume keeps local vector state between restarts. Remove it explicitly only when a clean data store is required.
