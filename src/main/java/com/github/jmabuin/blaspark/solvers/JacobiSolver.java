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

package com.github.jmabuin.blaspark.solvers;

import com.github.jmabuin.blaspark.io.IO;
import com.github.jmabuin.blaspark.operations.L1;
import com.github.jmabuin.blaspark.operations.L2;
import com.github.jmabuin.blaspark.operations.L3;
import com.github.jmabuin.blaspark.operations.OtherOperations;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;

/**
 * Class to implement the Conjugate Gradient method
 * @author Jose M. Abuin
 * @brief Class to perform the Conjugate Gradient method. Valid only for symmetric and positive definite matrices
 */
public class JacobiSolver {

	private static final Log LOG = LogFactory.getLog(JacobiSolver.class);
	private static final double EPSILON = 1.0e-5;

	/**
	 * We are going to perform Ax = b where A is the input matrix. x is the output vector and b is the input vector
	 * @param matrix The input matrix A
	 * @param inputVector The input vector b
	 * @param outputVector The output vector x
	 * @param numIterations The max number of iterations to perform
	 * @return The solution vector
	 */
	public static DenseVector solve(DistributedMatrix matrix, DenseVector inputVector, DenseVector outputVector,
									long numIterations, JavaSparkContext jsc) {

		long start = System.nanoTime();

		//Jacobi Method as shown in the example from https://en.wikipedia.org/wiki/Jacobi_method

		long 		k = 0;
		boolean		stop = false;
		int	    	i = 0;
		double		result	= 0.0;

		//Initial solution
		DenseVector x1;// = Vectors.zeros(inputVector.size()).toDense();
		double[] x1Values = new double[inputVector.size()];

		DenseVector x2 = Vectors.zeros(inputVector.size()).toDense();
		DenseVector res = Vectors.zeros(inputVector.size()).toDense();

		DistributedMatrix LU;
		DistributedMatrix Dinv;
		DistributedMatrix T = null;

		DenseVector C = Vectors.zeros(inputVector.size()).toDense();

		/*if(!isDiagonallyDominant(A,M,N,nz)){
			fprintf(stderr, "[%s] The matrix is not diagonally dominant\n",__func__);
			//return 0;
		}*/

		LU = OtherOperations.GetLU(matrix, jsc);

		Dinv = OtherOperations.GetD(matrix, true, jsc);

		for(i = 0; i< x1Values.length; i++){
			x1Values[i] = 1.0;
		}

		x1 = new DenseVector(x1Values);
		//T=-D^{-1}(L+U)
		T = L3.DGEMM(-1.0, Dinv, LU, 0.0, T, jsc);

		//C=D^{-1}B
		C = L2.DGEMV(1.0, Dinv, inputVector,0.0, C, jsc);

		long maxIterations = inputVector.size() * 2;

		if(numIterations != 0 ){
			maxIterations = numIterations;
		}

		while(!stop) {

			// x^{(1)}= Tx^{(0)}+C
			//x2 = T*x1
			x2 = L2.DGEMV(1.0, T, x1, 0.0, x2, jsc);

			//x2 = x2+C
			BLAS.axpy(1.0, C, x2);

			//res = A*x - b
			res = inputVector.copy();
			L2.DGEMV(1.0, matrix, x2, -1.0, res, jsc);

			result = L1.vectorSumElements(res);

			if((Math.abs(result)<=EPSILON)||(k == maxIterations)){
				//fprintf(stderr,"Sum vector res is %lg\n",result);
				stop = true;
			}

			x1 = x2.copy();

			k++;
		}

		outputVector = x2.copy();

		long end = System.nanoTime();

		LOG.warn("Total time in solve system is: "+(end - start)/1e9 + " and "+k+" iterations.");

		return outputVector;

	}

}
