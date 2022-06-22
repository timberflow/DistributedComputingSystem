package DSPPCode.spark.MLP.question;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.storage.StorageLevel;

import java.text.DecimalFormat;

public class ForwardStep{
  private static final DecimalFormat FORMAT = new DecimalFormat("#0.0000");
  public static class VectorSum implements Function2<double[][], double[][], double[][]> {
    @Override
    public double[][] call(double[][] a, double[][] b) throws Exception{
      double[][] out = new double[a.length][a[0].length];
      for (int i = 0; i < a.length; i++){
        for(int j = 0; j < a[0].length; j++)
          out[i][j] = a[i][j] + b[i][j];
      }
      return out;
    }
  }
  public BackwardWeights forward(double[][] weights1, double[][] weights2, JavaRDD<DataPoint> points){

    points.persist(StorageLevel.MEMORY_ONLY());

    ComputeMatrix computeMatrix = new ComputeMatrix();
    VectorSum vectorSum = new VectorSum();
    // int count = (int)points.count();
    int count = 150;
    double[][] cache1 = new double[count][weights1.length]; // B * D
    double[][] cache2 = new double[count][weights2.length]; // B * H

    JavaRDD<DataPoint> logits = points.map(
        point -> {
          double[][] x = new double[1][point.x.length];
          x[0] = point.x;
          x = computeMatrix.multiply(x, weights1,0);
          cache1[point.index] = point.x;
          x = computeMatrix.activate(x);
          cache2[point.index] = x[0];
          return new DataPoint(x[0], point.y, point.index);
        }
    ).map(
        point -> {
          double[][] x = new double[1][point.x.length];
          x[0] = point.x;
          x = computeMatrix.multiply(x, weights2,0);
          double[] line_score = computeMatrix.softmax(x[0]);
          return new DataPoint(line_score, point.y, point.index);
        }
    );


    int correct = (int)logits.filter(
        point -> {
          int best = 1;
          double best_logits = point.x[best];
          for (int i = 0; i < point.x.length; i++){
            if (best_logits < point.x[i]) best = i;
          }

          return (best == point.y);
        }
    ).count();

    /*
    int correct = logits.map(
        point -> {
          int best = 1;
          double best_logits = point.x[best];
          for (int i = 0; i < point.x.length; i++){
            if (best_logits < point.x[i]) best = i;
          }

          return (best == point.y ? 1:0);
        }
    ).reduce((i, j) -> (i + j));*/


    JavaRDD<Double> Loss = logits.map(
        point -> {
          double loss = computeMatrix.cross_entropy(point.x, point.y);
          return loss;
        }
    );
    double[][] grad2 = logits.map(
        point -> {
          double[] dout = computeMatrix.backward(point.x, point.y, count);
          double[][] grad = computeMatrix.multiply1D(cache2[point.index], dout);
          return grad;
        }
    ).reduce(vectorSum);

    double[][] grad1 = logits.map(
        point -> {
          double[][] dout = new double[1][weights2[0].length];
          dout[0] =  computeMatrix.backward(point.x, point.y, count);
          double[][] grad = computeMatrix.multiply(dout, weights2, 1);
          double[][] h1 = new double[1][weights2.length];
          h1[0] = computeMatrix.activate_backward(grad[0], cache2[point.index]);
          return computeMatrix.multiply1D(cache1[point.index], h1[0]);
        }
    ).reduce(vectorSum);

    double acc = (double) correct / (double) count;
    double loss = Loss.reduce((loss1, loss2) -> loss1 + loss2) / count;
    System.out.println("loss: " + FORMAT.format(loss) + " got " + correct + "/" + count
        + " correct acc: " + FORMAT.format(acc));

    return new BackwardWeights(grad1, grad2);

  }


}
