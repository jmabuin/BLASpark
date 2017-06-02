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

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import scala.Tuple2;

import java.util.List;

/**
 * @author Jose M. Abuin
 * @brief Class to perform Level 2 BLAS operations
 */
public class L2  {

    public static DenseVector DGEMV(DistributedMatrix matrix, DenseVector vector, JavaSparkContext jsc){

        // Case of IndexedRowMatrix
        if( matrix.getClass() == IndexedRowMatrix.class) {

            final Broadcast BC = jsc.broadcast(vector);

            IndexedRowMatrix indexedMatrix = (IndexedRowMatrix) matrix;

            JavaRDD<IndexedRow> rows = indexedMatrix.rows().toJavaRDD();
            List<Tuple2<Long, Double>> returnValues = rows.mapToPair(new PairFunction<IndexedRow, Long, Double>() {

                @Override
                public Tuple2<Long, Double> call(IndexedRow row) {
                    DenseVector vect = (DenseVector) BC.getValue();

                    return new Tuple2<Long, Double>(row.index(), BLAS.dot(row.vector(), vect));
                }

            }).collect();


            double[] stockArr = new double[returnValues.size()];

            //for(int i = 0; i< returnValues.size(); i++) {
            for(Tuple2<Long, Double> item : returnValues) {
                stockArr[item._1().intValue()] = item._2();
            }

            return new DenseVector(stockArr);
        }

        else {
            return null;
        }


    }

}
