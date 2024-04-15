import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class LinearRegressionTest {

    @Test
    public void testSlope() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 3, 4, 5, 6};
        LinearRegression lr = new LinearRegression(x, y);
        assertEquals(1, lr.slope(), 0.0001);
    }

    @Test
    public void testIntercept() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 3, 4, 5, 6};
        LinearRegression lr = new LinearRegression(x, y);
        assertEquals(1, lr.intercept(), 0.0001);
    }

    @Test
    public void testPredict() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 3, 4, 5, 6};
        LinearRegression lr = new LinearRegression(x, y);
        assertEquals(7, lr.predict(6), 0.0001);
    }
}