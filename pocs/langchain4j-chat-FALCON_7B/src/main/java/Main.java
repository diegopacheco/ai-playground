import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.huggingface.HuggingFaceChatModel;
import static dev.langchain4j.data.message.SystemMessage.systemMessage;
import static dev.langchain4j.data.message.UserMessage.userMessage;
import static dev.langchain4j.model.huggingface.HuggingFaceModelName.TII_UAE_FALCON_7B_INSTRUCT;
import static java.time.Duration.ofSeconds;

public class Main{
  public static void main(String args[]){
    HuggingFaceChatModel model = HuggingFaceChatModel.builder()
            .accessToken(System.getenv("HF_TOKEN"))
            .modelId(TII_UAE_FALCON_7B_INSTRUCT)
            .timeout(ofSeconds(30))
            .temperature(0.7)
            .maxNewTokens(20)
            .waitForModel(true)
            .build();

    AiMessage aiMessage = model.generate(
            systemMessage("You are a good friend of mine, who likes to answer with jokes"),
            userMessage("Hey Bro, what are you doing?")
    ).content();

    System.out.println(aiMessage.text());
  }
}
