package com.github.jmabuin.blaspark.solvers;

import com.github.jmabuin.blaspark.examples.ConjugateGradientExample;
import com.github.jmabuin.blaspark.operations.L1;
import com.github.jmabuin.blaspark.operations.L2;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;

/**
 * Created by chema on 2/16/17.
 */
public class ConjugateGradientSolver {

	private static final Log LOG = LogFactory.getLog(ConjugateGradientSolver.class);
	private static final double EPSILON = 1.0e-5;

	/**
	 * We are going to perform Ax = b where A is the input matrix. x is the output vector and b is the input vector
	 * @param matrix The input matrix A
	 * @param inputVector The input vector b
	 * @param outputVector The output vector x
	 * @param numIterations The max number of iterations to perform
	 * @return
	 */
	public static DenseVector solve(IndexedRowMatrix matrix, DenseVector inputVector, DenseVector outputVector,
									long numIterations, JavaSparkContext jsc) {


		if (numIterations == 0) {
			numIterations = inputVector.size() * 2;
		}

		boolean stop = false;

		long start = System.nanoTime();

		DenseVector r = inputVector.copy();

		//Fin -- r = b-A*x

		DenseVector Ap = null;

		//p=r
		DenseVector p = r.copy();

		//rsold = r*r
		//double rsold = L1.multiply(r,r);
		double rsold = BLAS.dot(r,r);

		double alpha = 0.0;

		double rsnew = 0.0;

		int k=0;

		while(!stop){



			//Inicio -- Ap=A*p
			Ap = L2.DGEMV(matrix, p, jsc);

			//Fin -- Ap=A*p

			//alpha=rsold/(p'*Ap)
			//alpha = rsold/L1.multiply(p,Ap);
			alpha = rsold/BLAS.dot(p,Ap);

			//x=x+alpha*p

			BLAS.axpy(alpha, p, outputVector);

			//r=r-alpha*Ap
			BLAS.axpy(-alpha, Ap, r);


			//rsnew = r'*r
			rsnew = BLAS.dot(r,r);


			if((Math.sqrt(rsnew)<=EPSILON)||(k >= (numIterations))){
				stop = true;
			}

			//p=r+rsnew/rsold*p
			BLAS.scal((rsnew/rsold),p);
			BLAS.axpy(1.0, r, p);


			/*
			LOG.info("JMAbuin ["+k+"]Current rsold is: "+rsold);
			LOG.info("JMAbuin ["+k+"]Current alpha is: "+alpha);
			LOG.info("JMAbuin ["+k+"]First Ap is: "+Ap.apply(0));
			LOG.info("JMAbuin ["+k+"]Cumsum Ap is: "+cumsum(Ap));
			LOG.info("JMAbuin ["+k+"]First P is: "+p.apply(0));
			LOG.info("JMAbuin ["+k+"]Cumsum P is: "+cumsum(p));
			LOG.info("JMAbuin ["+k+"]First X is: "+X.apply(0));
			LOG.info("JMAbuin ["+k+"]Cumsum X is: "+cumsum(X));
			LOG.info("JMAbuin ["+k+"]First R is: "+r.apply(0));
			LOG.info("JMAbuin ["+k+"]Cumsum R is: "+cumsum(r));
			LOG.info("JMAbuin ["+k+"]Current rsnew is: "+rsnew);
			*/


			rsold = rsnew;


			//LOG.info("JMAbuin ["+k+"]New rsold is: "+rsold);

			//runtime.gc();
			//long memory = runtime.totalMemory() - runtime.freeMemory();
			//System.out.println("Used memory iterarion "+k+" is megabytes: " + memory/(1024L * 1024L));
			k++;

		}


		//FIN GRADIENTE CONJUGADO

		long end = System.nanoTime();

		LOG.warn("Total time in solve system is: "+(end - start)/1e9 + " and "+k+" iterations.");

		return outputVector;

	}

}
