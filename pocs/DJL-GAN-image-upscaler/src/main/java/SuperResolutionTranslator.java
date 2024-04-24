import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class SuperResolutionTranslator implements Translator<Image, Image> {

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDManager manager = ctx.getNDManager();
        return new NDList(input.toNDArray(manager).toType(DataType.FLOAT32, false));
    }

    @Override
    public Image processOutput(TranslatorContext ctx, NDList list) {
        NDArray output = list.get(0).clip(0, 255);
        return ImageFactory.getInstance().fromNDArray(output.squeeze());
    }
}