package com.github.jmabuin.blaspark.examples.spark;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import scala.Tuple2;

/**
 * Created by chema on 5/31/17.
 */
public class Array2IndexedRow implements Function<Tuple2<Long, double[]>, IndexedRow> {

    @Override
    public IndexedRow call(Tuple2<Long, double[]> longTuple2) throws Exception {
        return new IndexedRow(longTuple2._1(), new DenseVector(longTuple2._2()));
    }
}
