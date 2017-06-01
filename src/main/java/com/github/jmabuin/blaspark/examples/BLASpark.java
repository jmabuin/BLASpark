package com.github.jmabuin.blaspark.examples;

import com.github.jmabuin.blaspark.examples.options.GeneralOptions;
import com.github.jmabuin.blaspark.io.IO;
import com.github.jmabuin.blaspark.io.RowPerLineInputFormat;
import com.github.jmabuin.blaspark.solvers.ConjugateGradientSolver;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import scala.Tuple2;

/**
 * Created by chema on 5/31/17.
 */
public class BLASpark {

    private static final Log LOG = LogFactory.getLog(BLASpark.class);

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf().setAppName("BLASpark - Example CG");

        sparkConf.set("spark.shuffle.reduceLocality.enabled","false");
        //sparkConf.set("spark.memory.useLegacyMode","true");


        //The ctx is created from the previous config
        JavaSparkContext ctx = new JavaSparkContext(sparkConf);
        //ctx.hadoopConfiguration().set("parquet.enable.summary-metadata", "false");

        GeneralOptions BLASparkOptions = new GeneralOptions(args);

        if(BLASparkOptions.getMode() == GeneralOptions.Mode.DMXV) {

            LOG.warn("Starting dense matrix dot vector multiplication example...");
            DmXV exDmXV = new DmXV(BLASparkOptions, ctx);

            exDmXV.calculate();
        }
        else if(BLASparkOptions.getMode() == GeneralOptions.Mode.CG) {

            LOG.warn("Starting conjugate gradient example...");
            ConjugateGradientExample exCG = new ConjugateGradientExample(BLASparkOptions, ctx);

            exCG.calculate();
        }
        else {
            LOG.warn("No execution mode selected...");
            BLASparkOptions.printHelp();
        }

    }

}
