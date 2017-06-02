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
 * Main class to perform the examples
 * @author Jose M. Abuin
 * @brief Main class to perform the examples
 */
public class BLASpark {

    private static final Log LOG = LogFactory.getLog(BLASpark.class);

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf();//.setAppName("BLASpark - Example CG");

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
