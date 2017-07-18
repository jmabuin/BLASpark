/**
 * Copyright 2017 José Manuel Abuín Mosquera <josemanuel.abuin@usc.es>
 *
 * This file is part of BLASpark.
 *
 * BLASpark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BLASpark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with BLASpark. If not, see <http://www.gnu.org/licenses/>.
 */

package com.github.jmabuin.blaspark.examples;

import com.github.jmabuin.blaspark.examples.options.GeneralOptions;
import com.github.jmabuin.blaspark.examples.spark.Array2IndexedRow;
import com.github.jmabuin.blaspark.io.IO;
import com.github.jmabuin.blaspark.io.RowPerLineInputFormat;
import com.github.jmabuin.blaspark.operations.L2;
import com.github.jmabuin.blaspark.operations.L3;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.*;

/**
 * Class to implement the dense matrix dot vector multiplication example
 * @author Jose M. Abuin
 * @brief Dense matrix dot vector multiplication example
 */
public class DmXDm {

    private static final Log LOG = LogFactory.getLog(DmXDm.class);

    private IndexedRowMatrix tmpMatrix;
    private IndexedRowMatrix tmpMatrix2;
    private DistributedMatrix matrix1;
    private DistributedMatrix matrix2;
    private DistributedMatrix outputMatrix;

    private String inputMatrixPath1;
    private String inputMatrixPath2;
    private String outputMatrixPath;

    private JavaSparkContext ctx;

    private double alpha;
    private double beta;

    public DmXDm(GeneralOptions DmXDm_Options, JavaSparkContext ctx ) {

        this.ctx = ctx;

        this.ctx.getConf().setAppName("BLASpark - Example DmXDm");

        long numPartitions = DmXDm_Options.getNumPartitions();

        this.inputMatrixPath2 = DmXDm_Options.getInputVectorPath();
        this.inputMatrixPath1 = DmXDm_Options.getInputMatrixPath();
        this.outputMatrixPath = DmXDm_Options.getOutputVectorPath();

        this.alpha = DmXDm_Options.getAlpha();
        this.beta = DmXDm_Options.getBeta();

        // Read MATRIX 1 input data
        JavaRDD<IndexedRow> inputMatrixData1;

        if(numPartitions != 0) {
            inputMatrixData1 = ctx.newAPIHadoopFile(inputMatrixPath1, RowPerLineInputFormat.class,
                    Long.class, double[].class, ctx.hadoopConfiguration())
                    .map(new Array2IndexedRow())
                    .repartition((int)numPartitions);
        }
        else {
            inputMatrixData1 = ctx.newAPIHadoopFile(inputMatrixPath1, RowPerLineInputFormat.class,
                    Long.class, double[].class, ctx.hadoopConfiguration())
                    .map(new Array2IndexedRow());
        }


        this.tmpMatrix = new IndexedRowMatrix(inputMatrixData1.rdd());

        if(DmXDm_Options.getMatrixFormat() == GeneralOptions.MatrixFormat.COORDINATE) {
            LOG.info("The matrix format is CoordinateMatrix");
            this.matrix1 = this.tmpMatrix.toCoordinateMatrix();
            ((CoordinateMatrix)this.matrix1).entries().cache();
        }
        else if(DmXDm_Options.getMatrixFormat() == GeneralOptions.MatrixFormat.BLOCK) {
            LOG.info("The matrix format is BlockMatrix. Nrows: "+DmXDm_Options.getRowsPerlBlock()+". Ncols: "+DmXDm_Options.getColsPerBlock());
            this.matrix1 = this.tmpMatrix.toBlockMatrix(DmXDm_Options.getRowsPerlBlock(), DmXDm_Options.getColsPerBlock());
            ((BlockMatrix)this.matrix1).blocks().cache();
        }
        else {
            //this.tmpMatrix.rows().cache();
            LOG.info("The matrix format is IndexedRowMatrix");
            this.matrix1 = this.tmpMatrix;
            ((IndexedRowMatrix)this.matrix1).rows().cache();
        }


        // Read MATRIX 2 input data
        JavaRDD<IndexedRow> inputMatrixData2;

        if(numPartitions != 0) {
            inputMatrixData2 = ctx.newAPIHadoopFile(inputMatrixPath2, RowPerLineInputFormat.class,
                    Long.class, double[].class, ctx.hadoopConfiguration())
                    .map(new Array2IndexedRow())
                    .repartition((int)numPartitions);
        }
        else {
            inputMatrixData2 = ctx.newAPIHadoopFile(inputMatrixPath2, RowPerLineInputFormat.class,
                    Long.class, double[].class, ctx.hadoopConfiguration())
                    .map(new Array2IndexedRow());
        }


        this.tmpMatrix2 = new IndexedRowMatrix(inputMatrixData2.rdd());

        if(DmXDm_Options.getMatrixFormat() == GeneralOptions.MatrixFormat.COORDINATE) {
            LOG.info("The matrix format is CoordinateMatrix");
            this.matrix2 = this.tmpMatrix2.toCoordinateMatrix();
            ((CoordinateMatrix)this.matrix2).entries().cache();
        }
        else if(DmXDm_Options.getMatrixFormat() == GeneralOptions.MatrixFormat.BLOCK) {
            LOG.info("The matrix format is BlockMatrix. Nrows: "+DmXDm_Options.getRowsPerlBlock()+". Ncols: "+DmXDm_Options.getColsPerBlock());
            this.matrix2 = this.tmpMatrix2.toBlockMatrix(DmXDm_Options.getRowsPerlBlock(), DmXDm_Options.getColsPerBlock());
            ((BlockMatrix)this.matrix2).blocks().cache();
        }
        else {
            //this.tmpMatrix.rows().cache();
            LOG.info("The matrix format is IndexedRowMatrix");
            this.matrix2 = this.tmpMatrix2;
            ((IndexedRowMatrix)this.matrix2).rows().cache();
        }


    }

    public void calculate() {

        //this.outputVector = ConjugateGradientSolver.solve(matrix, this.vector, this.outputVector, this.iterationNumber, this.ctx);
        this.outputMatrix = L3.DGEMM(this.alpha, this.matrix1, this.matrix2, this.beta, this.outputMatrix, this.ctx); //L2.DGEMV(this.alpha, this.matrix, this.vector, this.beta, this.outputVector, this.ctx);


        //IO.writeVectorToFileInHDFS(this.outputVectorPath, this.outputVector, this.ctx.hadoopConfiguration());
        IO.writeMatrixToFileInHDFS(this.outputMatrixPath, this.outputMatrix, this.ctx.hadoopConfiguration());

    }

}
