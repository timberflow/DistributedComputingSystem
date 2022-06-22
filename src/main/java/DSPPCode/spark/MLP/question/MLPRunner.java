package DSPPCode.spark.MLP.question;

import DSPPCode.spark.MLP.impl.IterationStepImpl;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.regex.Pattern;
import java.lang.Math;

/**
 * multiple layer perception
 */
public class MLPRunner {
    private static final DecimalFormat FORMAT = new DecimalFormat("#0.00");
    /**
     * 数据维度
     */
    private static int D;
    private static int H;
    private static int O;

    /**
     * 解析字符串生成DataPoint
     * 字符串[1 0 1 2 3 4 5 6 7 8 9 10]
     * y = 0
     * x = [1 2 3 4 5 6 7 8 9 10]
     * index = 1
     */
    public static class ParsePoint implements Function<String, DataPoint> {
        private static final Pattern SPACE = Pattern.compile(" ");

        @Override
        public DataPoint call(String line) throws Exception {
            String[] tok = line.split(" ");
            int index = Integer.parseInt(tok[0]);
            int y = Integer.parseInt(tok[1]);
            double[] x = new double[D];
            for (int i = 0; i < D; i++) {
              x[i] = Double.parseDouble(tok[i + 2]);
            }
            return new DataPoint(x, y, index);
        }
    }

    public static int run(String[] args) throws IOException {

        SparkSession spark = SparkSession
                .builder()
                .master("local")
                .appName("MLP")
                .getOrCreate();

        JavaRDD<String> lines = spark.read().textFile(args[0]).javaRDD();

        JavaRDD<DataPoint> points = lines.map(new ParsePoint());


        // 初始化权重默认是0

        String[] tok = args[2].split(" ");
        D = Integer.parseInt(tok[0]);
        H = Integer.parseInt(tok[1]);
        O = Integer.parseInt(tok[2]);

        double[][] w1 = new double[D][H];
        double[][] w2 = new double[H][O];
        for (int i = 0; i < D; i++){
          for(int j = 0; j < H; j++){
            w1[i][j] = 2 * (Math.random() - 0.5);
          }
        }
        for (int i = 0; i < H; i++){
          for(int j = 0; j < O; j++){
            w2[i][j] = 2 * (Math.random() - 0.5);
          }
        }
        // 计算迭代后的权重
        long start = System.currentTimeMillis();
        BackwardWeights bk = new IterationStepImpl().iteration(points, w1, w2);
        long end = System.currentTimeMillis();
        double span = (double) (end - start) / 1000.0;
        System.out.println("Complete in " + span + " s");

        w1 = bk.w1;
        w2 = bk.w2;
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(args[1])));

        bw.write("w1:\n");
        for (int i = 0; i < w1.length; i++) {
          for (int j = 0; j < w1[0].length; j++){
            bw.write(FORMAT.format(w1[i][j]) + " ");
          }
          bw.write("\n");
        }
        bw.write("w2:\n");
        for (int i = 0; i < w2.length; i++) {
          for (int j = 0; j < w2[0].length; j++){
            bw.write(FORMAT.format(w2[i][j]) + " ");
          }
          bw.write("\n");
        }
        bw.close();
        spark.stop();

        return 0;
    }

    public static void main(String[] args) throws IOException {
      run(args);
    }
}
