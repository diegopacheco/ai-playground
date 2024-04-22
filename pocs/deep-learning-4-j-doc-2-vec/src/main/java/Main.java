import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {
        // Set up document iterator
        String path = new File(".").getCanonicalPath();
        File dir = new File(path + "/data/");
        System.out.println("Path: " + dir.getAbsolutePath());

        DocumentIterator iter = new FileDocumentIterator(dir);
        // Set up tokenizer
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                return token.toLowerCase();
            }
        });

        // Create Word2Vec model
        Word2Vec word2Vec = new Word2Vec.Builder()
                .iterations(5)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(tokenizerFactory)
                .build();

        // Train model
        word2Vec.fit();

        // Save model
        WordVectorSerializer.writeWord2VecModel(word2Vec, path + "/model.bin");
        System.out.println("Model saved");

        // Load model
        Word2Vec loadedModel = WordVectorSerializer.readWord2VecModel(path + "/model.bin");
        System.out.println("Model loaded");

        // Get document vectors
        List<String> documents = readDocuments(dir);
        List<double[]> documentVectors = new ArrayList<>();
        for (String document : documents) {
            System.out.println("Adding document vector for: " + document);

            List<String> words = tokenizerFactory.create(document).getTokens();

            // Compute document vector as the average of word vectors
            double[] documentVector = new double[loadedModel.getLayerSize()];
            for (String word : words) {
                double[] wordVector = loadedModel.getWordVector(word);
                if (wordVector != null) {
                    for (int i = 0; i < documentVector.length; i++) {
                        documentVector[i] += wordVector[i];
                    }
                }
            }
            for (int i = 0; i < documentVector.length; i++) {
                documentVector[i] /= words.size();
            }

            documentVectors.add(documentVector);
        }
        System.out.println(documentVectors);
    }

    private static List<String> readDocuments(File dir){
        List<String> documents = new ArrayList<>();
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isFile()) {
                    try{
                        String content = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
                        documents.add(content);
                    }catch(Exception e){
                        throw new RuntimeException(e);
                    }
                }
            }
        }
        return documents;
    }
}