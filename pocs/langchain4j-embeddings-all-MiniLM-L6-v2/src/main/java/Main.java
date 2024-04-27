import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.huggingface.HuggingFaceEmbeddingModel;
import java.util.List;
import static dev.langchain4j.data.segment.TextSegment.textSegment;
import static java.util.Arrays.asList;

public class Main {
    public static void main(String args[]) {
        HuggingFaceEmbeddingModel model = HuggingFaceEmbeddingModel.builder()
                .accessToken(System.getenv("HF_TOKEN"))
                .modelId("sentence-transformers/all-MiniLM-L6-v2")
                .build();

        Embedding embedding = model.embed("hello").content();
        System.out.println("hello embeddings: " + embedding);

        List<Embedding> embeddings = model.embedAll(asList(
                textSegment("hello"),
                textSegment("hello world")
        )).content();

        System.out.println("hello embedding: " + embeddings.get(0).vector().length);
        System.out.println("hello world embedding: " + embeddings.get(1).vector().length);
    }
}