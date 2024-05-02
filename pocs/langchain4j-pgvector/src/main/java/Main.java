import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;
import org.testcontainers.utility.MountableFile;

import java.io.File;
import java.util.List;

public class Main {
    public static void main(String args[]) throws Exception {

        String path = new File(".").getPath();

        DockerImageName dockerImageName = DockerImageName.parse("pgvector/pgvector:pg16");
        try (PostgreSQLContainer<?> postgreSQLContainer = new PostgreSQLContainer<>(dockerImageName)
                                                                .withPassword("password")
                                                                .withDatabaseName("testdb")
                                                                .withEnv("POSTGRES_PASSWORD","password")
                                                                //.withEnv("POSTGRES_HOST_AUTH_METHOD","trust")
                                                                .withEnv("POSTGRES_DB","testdb")
                                                                .withEnv("POSTGRES_USER","testuser")
                                                                .withCopyFileToContainer(
                                                                        MountableFile.forClasspathResource("init.sql"),
                                                                        "/docker-entrypoint-initdb.d/init.sql"
                                                                )
                                                                .withConnectTimeoutSeconds(5000)) {
            postgreSQLContainer.start();

            EmbeddingStore<TextSegment> embeddingStore = PgVectorEmbeddingStore.builder()
                    .host(postgreSQLContainer.getHost())
                    .port(postgreSQLContainer.getFirstMappedPort())
                    .database("testdb") //postgreSQLContainer.getDatabaseName())
                    .user("testuser") // postgreSQLContainer.getUsername())
                    .password("password") //postgreSQLContainer.getPassword())
                    .table("test")
                    .dimension(384)
                    .build();

            Thread.sleep(3000); // to be sure that embeddings were persisted

            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            TextSegment segment1 = TextSegment.from("I like football.");
            Embedding embedding1 = embeddingModel.embed(segment1).content();
            embeddingStore.add(embedding1, segment1);

            TextSegment segment2 = TextSegment.from("The weather is good today.");
            Embedding embedding2 = embeddingModel.embed(segment2).content();
            embeddingStore.add(embedding2, segment2);

            Thread.sleep(1000); // to be sure that embeddings were persisted

            Embedding queryEmbedding = embeddingModel.embed("What is your favourite sport?").content();
            List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(queryEmbedding, 1);
            EmbeddingMatch<TextSegment> embeddingMatch = relevant.get(0);

            System.out.println(embeddingMatch.score()); // 0.8144289
            System.out.println(embeddingMatch.embedded().text()); // I like football.

        }
    }
}
