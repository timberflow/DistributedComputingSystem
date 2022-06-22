package DSPPCode.spark.MLP.question;

import org.apache.spark.api.java.JavaRDD;

import java.io.Serializable;


abstract public class IterationStep implements Serializable {

    /**
     * 梯度下降法修改权重的步长（学习率）
     * */
    public static double STEP = 0.10;
    /**
     * 训练周期数
     * */
    public static final int num_epochs = 100;

    /**
     *
     * @param bk  梯度
     * @param weights1 权重向量1
     * @param weights2 权重向量2
     * @return 利用梯度下降法迭代一次求出的权重向量
     */
    abstract public BackwardWeights runStep(double[][] weights1, double[][] weights2, BackwardWeights bk);
    /**

     */
    protected BackwardWeights iteration(JavaRDD<DataPoint> points, double[][] weights1, double[][] weights2) {

        BackwardWeights backwardWeights = new BackwardWeights(weights1, weights2);
        ForwardStep forwardStep = new ForwardStep();
        for (int i = 0; i < num_epochs; i++) {
            // lr decay
            if (i % 10 == 0 && i != 0){
              STEP *= 0.90;
            }
            System.out.println("epoch " + (i + 1) + ":");
            backwardWeights.set(forwardStep.forward(weights1, weights2, points));
            BackwardWeights tmp = runStep(weights1, weights2, backwardWeights);
            weights1 = tmp.w1;
            weights2 = tmp.w2;
        }
        return new BackwardWeights(weights1, weights2);
    }
}
