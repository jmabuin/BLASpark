package com.github.jmabuin.blaspark.examples;

import com.github.jmabuin.blaspark.examples.options.GeneralOptions;
import com.github.jmabuin.blaspark.examples.spark.Array2IndexedRow;
import com.github.jmabuin.blaspark.io.IO;
import com.github.jmabuin.blaspark.io.RowPerLineInputFormat;
import com.github.jmabuin.blaspark.operations.L2;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;

/**
 * Created by chema on 6/1/17.
 */
public class DmXV {

    private static final Log LOG = LogFactory.getLog(DmXV.class);

    private IndexedRowMatrix matrix;
    private DenseVector vector;
    private DenseVector outputVector;

    private String inputVectorPath;
    private String inputMatrixPath;
    private String outputVectorPath;

    private JavaSparkContext ctx;

    public DmXV(GeneralOptions DmXV_Options, JavaSparkContext ctx ) {

        this.ctx = ctx;

        this.ctx.getConf().setAppName("BLASpark - Example DmXV");

        long numPartitions = DmXV_Options.getNumPartitions();

        this.inputVectorPath = DmXV_Options.getInputVectorPath();
        this.inputMatrixPath = DmXV_Options.getInputMatrixPath();
        this.outputVectorPath = DmXV_Options.getOutputVectorPath();

        // Read MATRIX input data
        JavaRDD<IndexedRow> inputMatrixData;

        if(numPartitions != 0) {
            inputMatrixData = ctx.newAPIHadoopFile(inputMatrixPath, RowPerLineInputFormat.class,
                    Long.class, double[].class, ctx.hadoopConfiguration())
                    .map(new Array2IndexedRow())
                    .repartition((int)numPartitions);
        }
        else {
            inputMatrixData = ctx.newAPIHadoopFile(inputMatrixPath, RowPerLineInputFormat.class,
                    Long.class, double[].class, ctx.hadoopConfiguration())
                    .map(new Array2IndexedRow());
        }


        this.matrix = new IndexedRowMatrix(inputMatrixData.rdd());
        this.matrix.rows().cache();

        // Read VECTOR input data
        this.vector = IO.readVectorFromFileInHDFS(this.inputVectorPath, this.ctx.hadoopConfiguration());

        this.outputVector = Vectors.zeros(this.vector.size()).toDense();





    }

    public void calculate() {

        //this.outputVector = ConjugateGradientSolver.solve(matrix, this.vector, this.outputVector, this.iterationNumber, this.ctx);
        this.outputVector = L2.DGEMV(matrix, this.vector, this.ctx);

        IO.writeVectorToFileInHDFS(outputVectorPath, this.outputVector, this.ctx.hadoopConfiguration());

    }

}
