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
import com.github.jmabuin.blaspark.solvers.ConjugateGradientSolver;
import com.github.jmabuin.blaspark.solvers.JacobiSolver;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;

/**
 * @author Jose M. Abuin
 * @brief Class example of how the Conjugate Gradient solver can be used
 */
public class JacobiExample {

    private static final Log LOG = LogFactory.getLog(JacobiExample.class);

    private DistributedMatrix matrix;
    private DenseVector vector;
    private DenseVector outputVector;

    private String inputVectorPath;
    private String inputMatrixPath;
    private String outputVectorPath;

    private JavaSparkContext ctx;
    private long iterationNumber;

    private GeneralOptions.MatrixFormat matrixFormat;

    public JacobiExample(GeneralOptions Jacobi_Options, JavaSparkContext ctx ) {

        this.ctx = ctx;

        this.ctx.getConf().setAppName("BLASpark - Example jacobi");

        this.iterationNumber = Jacobi_Options.getIterationNumber();
        long numPartitions = Jacobi_Options.getNumPartitions();


        this.inputVectorPath = Jacobi_Options.getInputVectorPath();
        this.inputMatrixPath = Jacobi_Options.getInputMatrixPath();
        this.outputVectorPath = Jacobi_Options.getOutputVectorPath();

        // Read MATRIX input data
        this.matrixFormat = Jacobi_Options.getMatrixFormat();

        if(this.matrixFormat == GeneralOptions.MatrixFormat.PAIRLINE) {
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
            ((IndexedRowMatrix)this.matrix).rows().cache();

        }


        // Read VECTOR input data
        this.vector = IO.readVectorFromFileInHDFS(this.inputVectorPath, this.ctx.hadoopConfiguration());

        this.outputVector = Vectors.zeros(this.vector.size()).toDense();





    }

    public void calculate() {

        this.outputVector = JacobiSolver.solve(matrix, this.vector, this.outputVector, this.iterationNumber, this.ctx);

        IO.writeVectorToFileInHDFS(outputVectorPath, this.outputVector, this.ctx.hadoopConfiguration());

    }


}
