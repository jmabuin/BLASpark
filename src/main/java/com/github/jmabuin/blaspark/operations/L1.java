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

package com.github.jmabuin.blaspark.operations;

import org.apache.spark.mllib.linalg.DenseVector;

/**
 * This class sgould not be necessary, as L1 operations can be implemented with Spark functionalities from MLlib
 * @author Jose M. Abuin
 * @brief Class to perform Level 1 BLAS operations
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
