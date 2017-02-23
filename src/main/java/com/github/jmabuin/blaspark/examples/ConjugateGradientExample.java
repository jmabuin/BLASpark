package com.github.jmabuin.blaspark.examples;

import breeze.linalg.DenseMatrix;
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

	public static void main(String[] args) {

		SparkConf sparkConf = new SparkConf().setAppName("BLASpark - Example CG");

		sparkConf.set("spark.shuffle.reduceLocality.enabled","false");
		//sparkConf.set("spark.memory.useLegacyMode","true");

		// Kryo serializer
		/*sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");

		Class[] serializedClasses = {Location.class, Sketch.class};
		sparkConf.registerKryoClasses(serializedClasses);
		*/

		//The ctx is created from the previous config
		JavaSparkContext ctx = new JavaSparkContext(sparkConf);
		//ctx.hadoopConfiguration().set("parquet.enable.summary-metadata", "false");

		ConjugateGradientOptions CG_Options = new ConjugateGradientOptions(args);

		long iterationNumber = CG_Options.getIterationNumber();
		long numPartitions = CG_Options.getNumPartitions();


		String inputVectorPath = CG_Options.getInputVectorPath();
		String inputMatrixPath = CG_Options.getInputMatrixPath();
		String outputVectorPath = CG_Options.getOutputVectorPath();

		// Read MATRIX input data
		JavaRDD<IndexedRow> inputMatrixData;

		if(numPartitions != 0) {
			inputMatrixData = ctx.newAPIHadoopFile(inputMatrixPath, RowPerLineInputFormat.class,
					Long.class, double[].class, ctx.hadoopConfiguration()).map(new Function<Tuple2<Long, double[]>, IndexedRow>() {
				@Override
				public IndexedRow call(Tuple2<Long, double[]> longTuple2) throws Exception {
					return new IndexedRow(longTuple2._1(), new DenseVector(longTuple2._2()));
				}
			}).repartition((int)numPartitions);
		}
		else {
			inputMatrixData = ctx.newAPIHadoopFile(inputMatrixPath, RowPerLineInputFormat.class,
					Long.class, double[].class, ctx.hadoopConfiguration()).map(new Function<Tuple2<Long, double[]>, IndexedRow>() {
				@Override
				public IndexedRow call(Tuple2<Long, double[]> longTuple2) throws Exception {
					return new IndexedRow(longTuple2._1(), new DenseVector(longTuple2._2()));
				}
			});
		}


		IndexedRowMatrix matrix = new IndexedRowMatrix(inputMatrixData.rdd());
		matrix.rows().cache();

		// Read VECTOR input data
		DenseVector inputVector = IO.readVectorFromFileInHDFS(inputVectorPath, ctx.hadoopConfiguration());

		DenseVector outputVector = Vectors.zeros(inputVector.size()).toDense();


		outputVector = ConjugateGradientSolver.solve(matrix, inputVector, outputVector, iterationNumber, ctx);

		IO.writeVectorToFileInHDFS(outputVectorPath, outputVector, ctx.hadoopConfiguration());


	}


}
