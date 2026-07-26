package com.diegopacheco.s3vectors;

import java.util.List;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;
import software.amazon.awssdk.services.s3vectors.S3VectorsClient;
import software.amazon.awssdk.services.s3vectors.model.CreateIndexRequest;
import software.amazon.awssdk.services.s3vectors.model.CreateVectorBucketRequest;
import software.amazon.awssdk.services.s3vectors.model.DataType;
import software.amazon.awssdk.services.s3vectors.model.DeleteVectorsRequest;
import software.amazon.awssdk.services.s3vectors.model.DistanceMetric;
import software.amazon.awssdk.services.s3vectors.model.GetVectorsRequest;
import software.amazon.awssdk.services.s3vectors.model.PutInputVector;
import software.amazon.awssdk.services.s3vectors.model.PutVectorsRequest;
import software.amazon.awssdk.services.s3vectors.model.QueryVectorsRequest;
import software.amazon.awssdk.services.s3vectors.model.S3VectorsException;
import software.amazon.awssdk.services.s3vectors.model.VectorData;

@Service
public class VectorService {

    private final S3VectorsClient client;
    private final String bucketName;
    private final String indexName;
    private final int dimension;
    private volatile boolean ready;

    public VectorService(
            S3VectorsClient client,
            @Value("${s3-vectors.bucket-name}") String bucketName,
            @Value("${s3-vectors.index-name}") String indexName,
            @Value("${s3-vectors.dimension}") int dimension) {
        this.client = client;
        this.bucketName = bucketName;
        this.indexName = indexName;
        this.dimension = dimension;
    }

    public synchronized void ensureInfrastructure() {
        if (ready) {
            return;
        }
        createBucket();
        createIndex();
        ready = true;
    }

    public VectorResponse put(String key, List<Float> values) {
        validate(values);
        ensureInfrastructure();
        var vector = PutInputVector.builder()
                .key(key)
                .data(VectorData.fromFloat32(values))
                .build();
        client.putVectors(PutVectorsRequest.builder()
                .vectorBucketName(bucketName)
                .indexName(indexName)
                .vectors(vector)
                .build());
        return new VectorResponse(key, values);
    }

    public VectorResponse get(String key) {
        ensureInfrastructure();
        return client.getVectors(GetVectorsRequest.builder()
                        .vectorBucketName(bucketName)
                        .indexName(indexName)
                        .keys(key)
                        .returnData(true)
                        .build())
                .vectors()
                .stream()
                .map(vector -> new VectorResponse(vector.key(), vector.data().float32()))
                .findFirst()
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Vector not found"));
    }

    public List<SearchResponse> search(List<Float> values, int topK) {
        validate(values);
        ensureInfrastructure();
        return client.queryVectors(QueryVectorsRequest.builder()
                        .vectorBucketName(bucketName)
                        .indexName(indexName)
                        .queryVector(VectorData.fromFloat32(values))
                        .topK(topK)
                        .returnDistance(true)
                        .build())
                .vectors()
                .stream()
                .map(vector -> new SearchResponse(vector.key(), vector.distance()))
                .toList();
    }

    public void delete(String key) {
        ensureInfrastructure();
        client.deleteVectors(DeleteVectorsRequest.builder()
                .vectorBucketName(bucketName)
                .indexName(indexName)
                .keys(key)
                .build());
    }

    public String bucketName() {
        return bucketName;
    }

    public String indexName() {
        return indexName;
    }

    public int dimension() {
        return dimension;
    }

    private void createBucket() {
        try {
            client.createVectorBucket(CreateVectorBucketRequest.builder()
                    .vectorBucketName(bucketName)
                    .build());
        } catch (S3VectorsException exception) {
            ignoreConflict(exception);
        }
    }

    private void createIndex() {
        try {
            client.createIndex(CreateIndexRequest.builder()
                    .vectorBucketName(bucketName)
                    .indexName(indexName)
                    .dataType(DataType.FLOAT32)
                    .dimension(dimension)
                    .distanceMetric(DistanceMetric.COSINE)
                    .build());
        } catch (S3VectorsException exception) {
            ignoreConflict(exception);
        }
    }

    private void ignoreConflict(S3VectorsException exception) {
        if (exception.statusCode() != 409) {
            throw exception;
        }
    }

    private void validate(List<Float> values) {
        if (values.size() != dimension) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Vector dimension must be " + dimension);
        }
        if (values.stream().anyMatch(value -> !Float.isFinite(value))) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Vector values must be finite");
        }
        if (values.stream().allMatch(value -> value == 0.0f)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Cosine vectors cannot contain only zeros");
        }
    }
}
