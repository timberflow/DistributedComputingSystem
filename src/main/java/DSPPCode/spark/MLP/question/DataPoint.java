package DSPPCode.spark.MLP.question;

import java.io.Serializable;

/**
 * 数据点类
 */
public class DataPoint implements Serializable {
    public double[] x;
    public int y;
    public int index;

    DataPoint(double[] x, int y, int index) {
        this.x = x;
        this.y = y;
        this.index = index;
    }
}
