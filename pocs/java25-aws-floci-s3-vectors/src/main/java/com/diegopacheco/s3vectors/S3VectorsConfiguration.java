package com.diegopacheco.s3vectors;

import java.net.URI;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3vectors.S3VectorsClient;

@Configuration
public class S3VectorsConfiguration {

    @Bean
    S3VectorsClient s3VectorsClient(
            @Value("${s3-vectors.endpoint}") URI endpoint,
            @Value("${s3-vectors.region}") String region,
            @Value("${s3-vectors.access-key}") String accessKey,
            @Value("${s3-vectors.secret-key}") String secretKey) {
        return S3VectorsClient.builder()
                .endpointOverride(endpoint)
                .region(Region.of(region))
                .credentialsProvider(StaticCredentialsProvider.create(AwsBasicCredentials.create(accessKey, secretKey)))
                .build();
    }
}
