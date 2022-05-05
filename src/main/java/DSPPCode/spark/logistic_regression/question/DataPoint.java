package DSPPCode.spark.logistic_regression.question;

import java.io.Serializable;

/**
 * 数据点类
 */
public class DataPoint implements Serializable {
    public double[] x;
    public double y;

    DataPoint(double[] x, double y) {
        this.x = x;
        this.y = y;
    }
}
