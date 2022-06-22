package DSPPCode.spark.MLP.impl;

import DSPPCode.spark.MLP.question.BackwardWeights;
import DSPPCode.spark.MLP.question.IterationStep;

public class IterationStepImpl extends IterationStep{

  public BackwardWeights runStep(double[][] weights1, double[][] weights2, BackwardWeights bk) {
    double[][] next_weight1 = new double[weights1.length][weights1[0].length];
    double[][] next_weight2 = new double[weights2.length][weights2[0].length];
    for (int i = 0; i < next_weight1.length; i++) {
      for (int j = 0; j < next_weight1[0].length; j++){
        next_weight1[i][j] = weights1[i][j] - STEP * bk.w1[i][j];
      }
    }
    for (int i = 0; i < next_weight2.length; i++) {
      for (int j = 0; j < next_weight2[0].length; j++){
        next_weight2[i][j] = weights2[i][j] - STEP * bk.w2[i][j];
      }
    }
    return new BackwardWeights(next_weight1, next_weight2);
  }

}
