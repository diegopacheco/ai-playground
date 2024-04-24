import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public final class Main {

  private static final Logger logger = LoggerFactory.getLogger(Main.class);

  public static void main(String[] args) throws ModelException, TranslateException, IOException {
    String imagePath = "src/main/resources/";
    ImageFactory imageFactory = ImageFactory.getInstance();

    List<Image> inputImages =
            Collections.singletonList(imageFactory.fromFile(Paths.get(imagePath + "fox.png")));

    List<Image> enhancedImages = enhance(inputImages);

    logger.info("Using TensorFlow Engine. {} images generated.", enhancedImages.size());
    saveImages(inputImages, enhancedImages);
  }

  private static void saveImages(List<Image> input, List<Image> generated) throws IOException {
    Path outputPath = Paths.get("build/output/super-res/");
    Files.createDirectories(outputPath);

    save(generated, "image", outputPath);
    save(group(input, generated), "stitch", outputPath);

    logger.info("Generated images have been saved in: {}", outputPath);
  }

  private static void save(List<Image> images, String name, Path path) throws IOException {
    for (int i = 0; i < images.size(); i++) {
      Path imagePath = path.resolve(name + i + ".png");
      images.get(i).save(Files.newOutputStream(imagePath), "png");
    }
  }

  private static List<Image> group(List<Image> input, List<Image> generated) {
    NDList stitches = new NDList(input.size());

    try (NDManager manager = Engine.getEngine("TensorFlow").newBaseManager()) {
      for (int i = 0; i < input.size(); i++) {
        int width = 1024;
        int height = 1024;
        NDArray left = input.get(i).toNDArray(manager);
        NDArray right = generated.get(i).toNDArray(manager);

        // Maintain aspect ratio for the left image
        float originalWidth = input.get(i).getWidth();
        float originalHeight = input.get(i).getHeight();
        float aspectRatio = originalWidth / originalHeight;

        if (originalWidth > originalHeight) {
          // Landscape image
          height = (int) (width / aspectRatio);
        } else {
          // Portrait image
          width = (int) (height * aspectRatio);
        }

        left = NDImageUtils.resize(left, width, height, Image.Interpolation.BICUBIC);
        right = NDImageUtils.resize(right, 1024, 1024, Image.Interpolation.BICUBIC); // force upscale the right image to 1024x1024

        stitches.add(NDArrays.concat(new NDList(left, right), 1));
      }

      return stitches.stream()
              .map(array -> ImageFactory.getInstance().fromNDArray(array))
              .collect(Collectors.toList());
    }
  }

  public static List<Image> enhance(List<Image> inputImages)
          throws IOException, ModelException, TranslateException {

    String modelUrl =
            "https://storage.googleapis.com/tfhub-modules/captain-pool/esrgan-tf2/1.tar.gz";
    Criteria<Image, Image> criteria =
            Criteria.builder()
                    .optApplication(Application.CV.IMAGE_ENHANCEMENT)
                    .setTypes(Image.class, Image.class)
                    .optModelUrls(modelUrl)
                    .optOption("Tags", "serve")
                    .optOption("SignatureDefKey", "serving_default")
                    .optTranslator(new SuperResolutionTranslator())
                    .optEngine("TensorFlow")
                    .optProgress(new ProgressBar())
                    .build();

    try (ZooModel<Image, Image> model = criteria.loadModel();
         Predictor<Image, Image> enhancer = model.newPredictor()) {
      return enhancer.batchPredict(inputImages);
    }
  }
}