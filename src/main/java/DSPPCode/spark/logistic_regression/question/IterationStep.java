package DSPPCode.spark.logistic_regression.question;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

import java.io.Serializable;


abstract public class IterationStep implements Serializable {

    /**
     * 梯度下降法修改权重的步长（学习率）
     * */
    public static final int STEP = 1;
    /**
     * 终止条件阈值
     * */
    public static final double THRESHOLD = 0.01;
    /**
     * TODO:终止条件，当新的权重和旧的权重平方距离(欧式距离的平方)小小于阈值（0.01）时迭代终止
     *
     * @param newWeight 新的权重
     * @param old       上次迭代的权重
     */
    abstract public boolean termination(double[] old, double[] newWeight);

    /**
     * TODO：利用此类中给的工具类，根据梯度下降法求解梯度
     *
     * @param points  数据点
     * @param weights 权重向量
     * @return 利用梯度下降法迭代一次求出的权重向量
     */
    abstract public double[] runStep(JavaRDD<DataPoint> points, double[] weights);

    /**
     * 迭代逻辑
     *
     * @param points  数据点
     * @param weights 权重
     * @return 利用梯度下降法多次迭代求出的最终权重向量
     */
    protected double[] iteration(JavaRDD<DataPoint> points, double[] weights) {

        double[] old = new double[weights.length];
        old[0] = 10.0;
        boolean sign = false;
        while (!termination(old, weights)) {
            if (sign) {
                old = weights.clone();
            } else {
                sign = true;
            }
            weights = runStep(points, weights);
        }
        return weights;
    }

    /**
     * TODO：实现向量求和
     *
     * 向量求和方法，通常由reduce算子调用
     * result[i] = a[i] + b[i]
     */
    abstract public static class VectorSum implements Function2<double[], double[], double[]> {
        @Override
        abstract public double[] call(double[] a, double[] b) throws Exception;
    }

    /**
     * TODO：根据readme所给公式求梯度
     *
     * 根据公式计算每个点的梯度
     */
    abstract public class ComputeGradient implements Function<DataPoint, double[]> {
        public final double[] weights;

        public ComputeGradient(double[] weights) {
            this.weights = weights;
        }

        @Override
        abstract public double[] call(DataPoint dataPoint) throws Exception;
    }

}
