package com.github.jmabuin.blaspark.operations;

import org.apache.spark.mllib.linalg.DenseVector;

/**
 * Created by chema on 2/16/17.
 */
public class L1 {

	public static double multiply(DenseVector v1, DenseVector v2) {

		double result = 0;

		for( int i = 0; i< v1.size(); i++){

			result = result + v1.apply(i) * v2.apply(i);

		}

		return result;

	}

}
