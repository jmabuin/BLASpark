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
public class DmXV {

    private static final Log LOG = LogFactory.getLog(DmXV.class);

    private IndexedRowMatrix tmpMatrix;
    private DistributedMatrix matrix;
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


        this.tmpMatrix = new IndexedRowMatrix(inputMatrixData.rdd());

        if(DmXV_Options.getMatrixFormat() == GeneralOptions.MatrixFormat.COORDINATE) {
            LOG.info("The matrix format is CoordinateMatrix");
            this.matrix = this.tmpMatrix.toCoordinateMatrix();
            ((CoordinateMatrix)this.matrix).entries().cache();
        }
        else if(DmXV_Options.getMatrixFormat() == GeneralOptions.MatrixFormat.BLOCK) {
            LOG.info("The matrix format is BlockMatrix. Nrows: "+DmXV_Options.getRowsPerlBlock()+". Ncols: "+DmXV_Options.getColsPerBlock());
            this.matrix = this.tmpMatrix.toBlockMatrix(DmXV_Options.getRowsPerlBlock(), DmXV_Options.getColsPerBlock());
            ((BlockMatrix)this.matrix).blocks().cache();
        }
        else {
            //this.tmpMatrix.rows().cache();
            LOG.info("The matrix format is IndexedRowMatrix");
            this.matrix = this.tmpMatrix;
            ((IndexedRowMatrix)this.matrix).rows().cache();
        }


        // Read VECTOR input data
        this.vector = IO.readVectorFromFileInHDFS(this.inputVectorPath, this.ctx.hadoopConfiguration());

        this.outputVector = Vectors.zeros(this.vector.size()).toDense();





    }

    public void calculate() {

        //this.outputVector = ConjugateGradientSolver.solve(matrix, this.vector, this.outputVector, this.iterationNumber, this.ctx);
        this.outputVector = L2.DGEMV(this.matrix, this.vector, this.ctx);

        IO.writeVectorToFileInHDFS(outputVectorPath, this.outputVector, this.ctx.hadoopConfiguration());

    }

}
