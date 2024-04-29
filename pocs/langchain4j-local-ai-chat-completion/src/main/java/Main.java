import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.localai.LocalAiChatModel;

public class Main {
    public static void main(String args[]) {
        ChatLanguageModel model = LocalAiChatModel.builder()
                .baseUrl("http://localhost:8080")
                .modelName("gpt-4")
                .temperature(0.9)
                .build();
        String answer = model.generate("How are you?");
        System.out.println(answer);
    }
}
