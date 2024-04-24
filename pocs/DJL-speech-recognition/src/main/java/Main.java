import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.audio.Audio;
import ai.djl.modality.audio.AudioFactory;
import ai.djl.modality.audio.translator.SpeechRecognitionTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public final class Main {

    public static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws Exception {
        logger.info("Result: {}", predict());
    }

    public static String predict() throws IOException, ModelException, TranslateException {
        // Load model.
        // Wav2Vec2 model is a speech model that accepts a float array corresponding to the raw
        // waveform of the speech signal.
        String url = "https://resources.djl.ai/test-models/pytorch/wav2vec2.zip";
        Criteria<Audio, String> criteria =
                Criteria.builder()
                        .setTypes(Audio.class, String.class)
                        .optModelUrls(url)
                        .optTranslatorFactory(new SpeechRecognitionTranslatorFactory())
                        .optModelName("wav2vec2.ptl")
                        .optEngine("PyTorch")
                        .build();

        // Read in audio file
        String wave = "https://resources.djl.ai/audios/speech.wav";
        Audio audio = AudioFactory.newInstance().fromUrl(wave);
        try (ZooModel<Audio, String> model = criteria.loadModel();
             Predictor<Audio, String> predictor = model.newPredictor()) {
            return predictor.predict(audio);
        }
    }
}