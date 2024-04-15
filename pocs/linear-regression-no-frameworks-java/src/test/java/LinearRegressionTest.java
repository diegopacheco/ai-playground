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
        assertEquals(11, lr.predict(10), 0.0001);
        assertEquals(100, lr.predict(99), 0.0001);
    }


    @Test
    public void testR2() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 3, 4, 5, 6};
        LinearRegression lr = new LinearRegression(x, y);
        assertEquals(1, lr.r2(), 0.0001);
    }

    @Test
    public void testInterceptStdErr() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 3, 4, 5, 6};
        LinearRegression lr = new LinearRegression(x, y);
        assertEquals(0, lr.interceptStdErr(), 0.0001);
    }

    @Test
    public void testSlopeStdErr() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 3, 4, 5, 6};
        LinearRegression lr = new LinearRegression(x, y);
        assertEquals(0, lr.slopeStdErr(), 0.0001);
    }
}