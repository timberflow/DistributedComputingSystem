package DSPPCode.spark.MLP.question;

public class BackwardWeights{
  public double[][] w1;
  public double[][] w2;

  public BackwardWeights(double[][] w1, double[][] w2){
    this.w1 = w1;
    this.w2 = w2;
  }

  public void set(BackwardWeights bk){
    this.w1 = bk.w1;
    this.w2 = bk.w2;
  }

}
