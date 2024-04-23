import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public final class BertQaInference {

    private static final Logger logger = LoggerFactory.getLogger(BertQaInference.class);

    private BertQaInference() {}

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        String answer = BertQaInference.predict();
        logger.info("Answer: {}", answer);
    }

    public static String predict() throws IOException, TranslateException, ModelException {
        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                + "Which operated between December 2004 and April 2006. "
                + "It ceased operations after its Japanese distributor folded.";

        QAInput input = new QAInput(question, paragraph);
        logger.info("Paragraph: {}", input.getParagraph());
        logger.info("Question: {}", input.getQuestion());

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.QUESTION_ANSWER)
                        .setTypes(QAInput.class, String.class)
                        //.optFilter("backbone", "bert")
                        .optEngine(Engine.getDefaultEngineName())
                        .optProgress(new ProgressBar())
                        .optModelName("bert-large-uncased-whole-word-masking-finetuned-squad")
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel()) {
            try (Predictor<QAInput, String> predictor = model.newPredictor()) {
                return predictor.predict(input);
            }
        }
    }
}