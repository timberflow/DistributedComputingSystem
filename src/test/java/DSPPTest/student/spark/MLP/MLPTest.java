package DSPPTest.student.spark.MLP;


import DSPPCode.spark.MLP.question.MLPRunner;
import DSPPTest.student.TestTemplate;
import org.junit.Test;

import static DSPPTest.util.FileOperator.deleteFolder;
import static DSPPTest.util.Verifier.verifyKV;

/**
 * @author chenqh
 * @version 1.0.0
 * @date 2019-12-28
 */
public class MLPTest extends TestTemplate {
    /**
     * 测试结果
     */
    @Test(timeout = 18000)
    public void testResult() throws Exception {
        String inputFile = root + "/spark/logistic_regression/input";
        String outputFile = root + "/spark/logistic_regression/output";
        String answerFile = root + "/spark/logistic_regression/answer";
        String[] args = new String[3];
        args[0] = inputFile;
        args[1] = outputFile;
        args[2] = "4 10 3";
        // 删除旧输出
        deleteFolder(outputFile);

        MLPRunner.run(args);

        System.out.println("恭喜通过~");
    }
}
