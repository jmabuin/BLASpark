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

package com.github.jmabuin.blaspark.examples.spark;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import scala.Tuple2;

/**
 * Class that implements the Spark function to convert a double array to an IndexedRow
 * @author Jose M. Abuin
 * @brief Class that implements the Spark function to convert a double array to an IndexedRow
 */
public class Array2IndexedRow implements Function<Tuple2<Long, double[]>, IndexedRow> {

    @Override
    public IndexedRow call(Tuple2<Long, double[]> longTuple2) throws Exception {
        return new IndexedRow(longTuple2._1(), new DenseVector(longTuple2._2()));
    }
}
