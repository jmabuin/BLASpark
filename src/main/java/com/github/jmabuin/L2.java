package com.github.jmabuin;

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
 * Created by jabuinmo on 07.02.17.
 */
public class L2  {


    public static DenseVector DGEMV(DistributedMatrix matrix, DenseVector vector, JavaSparkContext jsc){

        // Case of IndexedRowMatrix
        if( matrix.getClass() == IndexedRowMatrix.class) {

            final Broadcast BC = jsc.broadcast(vector);

            IndexedRowMatrix indexedMatrix = (IndexedRowMatrix) matrix;

            JavaRDD<IndexedRow> rows = indexedMatrix.rows().toJavaRDD();
            List<Double> returnValues = rows.mapToPair(new PairFunction<IndexedRow, Long, Double>() {

                @Override
                public Tuple2<Long, Double> call(IndexedRow row) {
                    DenseVector vect = (DenseVector) BC.getValue();

                    return new Tuple2<Long, Double>(row.index(), BLAS.dot(row.vector(), vect));
                }

            }).sortByKey().values().collect();


            double[] stockArr = new double[returnValues.size()];

            for(int i = 0; i< returnValues.size(); i++) {
                stockArr[i] = returnValues.get(i);
            }

            return new DenseVector(stockArr);
        }

        else {
            return null;
        }


    }

}
