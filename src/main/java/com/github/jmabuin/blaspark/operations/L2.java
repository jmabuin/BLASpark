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

package com.github.jmabuin.blaspark.operations;

import com.github.jmabuin.blaspark.spark.MatrixEntriesMultiplication;
import com.github.jmabuin.blaspark.spark.MatrixEntriesMultiplicationReducer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.*;
import scala.Tuple2;

import java.util.Iterator;
import java.util.List;

/**
 * @author Jose M. Abuin
 * @brief Class to perform Level 2 BLAS operations
 */
public class L2  {

    private static final Log LOG = LogFactory.getLog(L2.class);

    // y := alpha*A*x + beta*y
    public static DenseVector DGEMV(double alpha, DistributedMatrix A, DenseVector x, double beta, DenseVector y, JavaSparkContext jsc){

        // First form  y := beta*y.
        if (beta != 1.0) {
            if (beta == 0.0) {
                y = Vectors.zeros(y.size()).toDense();
            }
            else {
                BLAS.scal(beta, y);
            }
        }

        if (alpha == 0.0) {
            return y;
        }

        DenseVector tmpVector = Vectors.zeros(y.size()).toDense();

        // Form  y := alpha*A*x + y.
        // Case of IndexedRowMatrix
        if( A.getClass() == IndexedRowMatrix.class) {
            tmpVector = L2.DGEMV_IRW((IndexedRowMatrix) A, alpha, x, jsc);
        }
        else if (A.getClass() == CoordinateMatrix.class) {
            tmpVector = L2.DGEMV_COORD((CoordinateMatrix) A, alpha, x, jsc);
        }
        else if (A.getClass() == BlockMatrix.class){
            tmpVector = L2.DGEMV_BCK((BlockMatrix) A, alpha, x, jsc);
        }
        else {
            tmpVector = null;
        }

        BLAS.axpy(1.0, tmpVector, y);


        return y;

    }

    private static DenseVector DGEMV_IRW(IndexedRowMatrix matrix, double alpha, DenseVector vector, JavaSparkContext jsc) {

        final Broadcast BC = jsc.broadcast(vector);
        final Broadcast<Double> AlphaBC = jsc.broadcast(alpha);

        //IndexedRowMatrix indexedMatrix = (IndexedRowMatrix) matrix;

        JavaRDD<IndexedRow> rows = matrix.rows().toJavaRDD();
        List<Tuple2<Long, Double>> returnValues = rows.mapToPair(new PairFunction<IndexedRow, Long, Double>() {

            @Override
            public Tuple2<Long, Double> call(IndexedRow row) {
                DenseVector vect = (DenseVector) BC.getValue();
                double alphaBCRec = AlphaBC.getValue().doubleValue();

                DenseVector tmp = row.vector().copy().toDense();

                BLAS.scal(alphaBCRec, tmp);

                return new Tuple2<Long, Double>(row.index(), BLAS.dot(tmp, vect));
            }

        }).collect();


        double[] stockArr = new double[returnValues.size()];

        //for(int i = 0; i< returnValues.size(); i++) {
        for(Tuple2<Long, Double> item : returnValues) {
            stockArr[item._1().intValue()] = item._2();
        }

        return new DenseVector(stockArr);
    }

    private static DenseVector DGEMV_COORD(CoordinateMatrix matrix, double alpha, DenseVector vector, JavaSparkContext jsc) {

        JavaRDD<MatrixEntry> items = matrix.entries().toJavaRDD();
        DenseVector result = items.mapPartitions(new MatrixEntriesMultiplication(vector, alpha))
                .reduce(new MatrixEntriesMultiplicationReducer());

        return result;
    }

    /**
     * TODO: Not working in Spark 2.1.0 nor 1.6.1 because of the toBlockMatrix method (I think)
     * @param matrix
     * @param vector
     * @param jsc
     * @return
     */
    private static DenseVector DGEMV_BCK(BlockMatrix matrix, double alpha, DenseVector vector, JavaSparkContext jsc) {

        final Broadcast BC = jsc.broadcast(vector);
        //final Broadcast AlphaBC = jsc.broadcast(alpha);
        final Broadcast<Double> AlphaBC = jsc.broadcast(alpha);
        // Theoretically, the index should be a Tuple2<Integer, Integer>
        JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> blocks = matrix.blocks().toJavaRDD();
        //JavaRDD<Tuple2<Tuple2<Integer, Integer>, Matrix>> blocks = matrix.blocks().toJavaRDD();

        DenseVector returnValues = blocks.map(new Function<Tuple2<Tuple2<Object,Object>,Matrix>, DenseVector>() {

            @Override
            public DenseVector call(Tuple2<Tuple2<Object, Object>, Matrix> block) {
                LOG.warn("[JMAbuin] Entering Map Phase");
                DenseVector inputVect = (DenseVector) BC.getValue();
                double alphaBCRec = AlphaBC.getValue().doubleValue();

                LOG.warn("[JMAbuin] Vector items: "+inputVect.size());

                double finalResultArray[] = new double[inputVect.size()];

                for(int i = 0; i< finalResultArray.length; i++) {
                    finalResultArray[i] = 0.0;
                }

                LOG.warn("[JMAbuin] Before loading rows and cols: "+inputVect.size());
                Integer row = (Integer)block._1._1; //Integer.parseInt(block._1._1.toString());
                Integer col = (Integer)block._1._2;//Integer.parseInt(block._1._2.toString());

                LOG.warn("[JMAbuin] Row is: "+row);
                LOG.warn("[JMAbuin] Col is: "+col);

                Matrix matr = block._2;

                double vectValues[] = new double[matr.numCols()];
                double resultArray[] = new double[matr.numCols()];

                for(int i = col; i < matr.numCols();i++) {
                    vectValues[(i-col)] = inputVect.apply(i);
                    resultArray[(i-col)] = 0.0;
                }

                DenseVector result = Vectors.zeros(matr.numCols()).toDense();//new DenseVector(resultArray);

                BLAS.gemv(alphaBCRec, matr, new DenseVector(vectValues), 0.0, result);

                for(int i = col; i < matr.numCols();i++) {
                    finalResultArray[i] = result.apply((i-col));
                }

                return new DenseVector(finalResultArray);
            }

        }).reduce(new Function2<DenseVector, DenseVector, DenseVector>() {
            @Override
            public DenseVector call(DenseVector vector, DenseVector vector2) throws Exception {
                double result[] = new double[vector.size()];

                for(int i = 0; i< result.length;i++) {
                    result[i] = vector.apply(i) + vector2.apply(i);
                }

                return new DenseVector(result);
            }
        });


        return returnValues;
    }

}
