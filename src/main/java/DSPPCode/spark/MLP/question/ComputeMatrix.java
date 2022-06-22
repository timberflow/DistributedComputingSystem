package DSPPCode.spark.MLP.question;

import scala.Serializable;
import java.lang.Math;
import java.text.DecimalFormat;

public class ComputeMatrix implements Serializable {

  private DecimalFormat FORMAT = new DecimalFormat("#0.00");

  public void display(double[][] A){
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++){
        System.out.print(FORMAT.format(A[i][j]) + " ");
      }
      System.out.println();
    }
  }

  public double[][] multiply(double[][] A, double[][] B, int transpose){
    int I = A.length;
    int K = A[0].length;
    if (transpose == 1){
      int J = B.length;
      double[][] result = new double[I][J];
      for (int i = 0; i < I; i++){
        for (int j = 0; j < J; j++){
          for(int k = 0; k < K; k++){
            result[i][j] += A[i][k] * B[j][k];
          }

        }
      }
      return result;
    } else {
      int J = B[0].length;
      double[][] result = new double[I][J];
      for (int i = 0; i < I; i++){
        for (int j = 0; j < J; j++){
          for(int k = 0; k < K; k++){
            result[i][j] += A[i][k] * B[k][j];
          }
        }
      }
      return result;
    }

  }

  public double[][] multiply1D(double[] a, double[] b){
    int I = a.length;
    int J = b.length;
    double[][] result = new double[I][J];
    for (int i = 0; i < I; i++){
      for (int j = 0; j < J; j++){
        result[i][j] = a[i] * b[j];
      }
    }
    return result;
  }

  public double[][] activate(double[][] A){
    for (int i = 0; i < A.length; i++){
      for (int j = 0; j < A[0].length; j++){
        A[i][j] = Math.tanh(A[i][j]);
      }
    }
    return A;
  }

  public double[] activate_backward(double[] x, double[] c){
    for (int i = 0; i < x.length; i++){
      x[i] = x[i] * (1 - c[i] * c[i]);
    }
    return x;
  }

  public double[] softmax(double[] x){
    double[] prob = new double[x.length];
    double sum = 0;
    for (int i = 0; i < x.length; i++){
      sum += Math.exp(x[i]);
    }
    for (int i = 0; i < x.length; i++){
      prob[i] = Math.exp(x[i]) / sum;
    }
    return prob;
  }

  public double cross_entropy(double[] score, int y){
    return -Math.log(score[y]);
  }

  public double[] backward(double[] prob, int y, int n){
    prob[y] -= 1;
    for (int i = 0; i < prob.length; i++)prob[i] /= n;
    return prob;
  }

}
