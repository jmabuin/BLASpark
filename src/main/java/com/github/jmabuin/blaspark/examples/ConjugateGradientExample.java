package com.github.jmabuin.blaspark.examples;

import breeze.linalg.DenseMatrix;
import com.github.jmabuin.blaspark.examples.options.GeneralOptions;
import com.github.jmabuin.blaspark.examples.spark.Array2IndexedRow;
import com.github.jmabuin.blaspark.io.IO;
import com.github.jmabuin.blaspark.io.RowPerLineInputFormat;
import com.github.jmabuin.blaspark.solvers.ConjugateGradientSolver;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import scala.Tuple2;

/**
 * Created by chema on 2/16/17.
 */
public class ConjugateGradientExample {

	private static final Log LOG = LogFactory.getLog(ConjugateGradientExample.class);

	private IndexedRowMatrix matrix;
	private DenseVector vector;
	private DenseVector outputVector;

	private String inputVectorPath;
	private String inputMatrixPath;
	private String outputVectorPath;

	private JavaSparkContext ctx;
	private long iterationNumber;

	public ConjugateGradientExample(GeneralOptions CG_Options, JavaSparkContext ctx ) {

		this.ctx = ctx;

		this.ctx.getConf().setAppName("BLASpark - Example CG");

		this.iterationNumber = CG_Options.getIterationNumber();
		long numPartitions = CG_Options.getNumPartitions();


		this.inputVectorPath = CG_Options.getInputVectorPath();
		this.inputMatrixPath = CG_Options.getInputMatrixPath();
		this.outputVectorPath = CG_Options.getOutputVectorPath();

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


		this.matrix = new IndexedRowMatrix(inputMatrixData.rdd());
		this.matrix.rows().cache();

		// Read VECTOR input data
		this.vector = IO.readVectorFromFileInHDFS(this.inputVectorPath, this.ctx.hadoopConfiguration());

		this.outputVector = Vectors.zeros(this.vector.size()).toDense();





	}

	public void calculate() {

		this.outputVector = ConjugateGradientSolver.solve(matrix, this.vector, this.outputVector, this.iterationNumber, this.ctx);

		IO.writeVectorToFileInHDFS(outputVectorPath, this.outputVector, this.ctx.hadoopConfiguration());

	}


}
