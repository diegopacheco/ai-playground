import dev.langchain4j.model.huggingface.HuggingFaceLanguageModel;

import static dev.langchain4j.model.huggingface.HuggingFaceModelName.TII_UAE_FALCON_7B_INSTRUCT;
import static java.time.Duration.ofSeconds;

public class Main{
  public static void main(String args[]){
    HuggingFaceLanguageModel model = HuggingFaceLanguageModel.builder()
            .accessToken(System.getenv("HF_TOKEN"))
            .modelId(TII_UAE_FALCON_7B_INSTRUCT)
            .timeout(ofSeconds(15))
            .temperature(0.7)
            .maxNewTokens(20)
            .waitForModel(true)
            .build();

    String answer = model.generate("What is the capital of the USA?").content();
    System.out.println(answer);

    System.out.println(model.generate("What is the capital of Brazil?").content());

    System.out.println(model.generate("Conan: What is best in life?").content());
  }
}
