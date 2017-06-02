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

package com.github.jmabuin.blaspark.spark;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;

/**
 * Created by chema on 6/2/17.
 */
public class MatrixEntriesMultiplicationReducer implements Function2<DenseVector, DenseVector, DenseVector> {

    @Override
    public DenseVector call(DenseVector vector, DenseVector vector2) throws Exception {
        BLAS.axpy(1.0, vector, vector2);

        return vector2;
    }
}
