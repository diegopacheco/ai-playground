import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.File;
import java.nio.file.Paths;
import java.util.Arrays;

public class ImagePredictor {

    // Construct neural network
    private static Block block =
            new Mlp(
                    Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                    Mnist.NUM_CLASSES,
                    new int[]{128, 64});

    public static void main(String[] args) throws Exception {

        System.out.println("F.Y.I the number is 5 - lets see is AI knows... ");

        try (NDManager manager = NDManager.newBaseManager()) {
            String basePath = new File(".").getCanonicalPath();
            String imagePath = basePath + "/src/main/resources/number.png";

            Image img = ImageFactory.getInstance().fromFile(Paths.get(imagePath));
            NDList list = null;

            Pipeline pipeline = new Pipeline();
            pipeline.add(new Resize(28, 28));
            pipeline.add(new ToTensor());

            NDArray array = img.toNDArray(manager);
            NDList arrayToList = new NDList(array);
            list = pipeline.transform(arrayToList);

            try (Model model = Model.newInstance("mlp")) {
                model.setBlock(block);
                model.load(Paths.get(basePath+"/mlp_model/"));

                Predictor<NDList, Classifications> predictor = model.newPredictor(new MyTranslator());
                Classifications classifications = predictor.predict(list);
                System.out.println(classifications);
            }
        }
    }

    static class MyTranslator implements Translator<NDList, Classifications> {

        @Override
        public NDList processInput(TranslatorContext ctx, NDList input) {
            Pipeline pipeline = new Pipeline();
            pipeline.add(new Resize(28, 28));
            pipeline.add(new ToTensor());
            return pipeline.transform(input);
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, ai.djl.ndarray.NDList list) {
            return new Classifications(Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"), list.singletonOrThrow());
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}