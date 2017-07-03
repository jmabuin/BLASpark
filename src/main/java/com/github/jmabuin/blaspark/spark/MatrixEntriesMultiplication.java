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

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by chema on 6/2/17.
 */
public class MatrixEntriesMultiplication implements FlatMapFunction<Iterator<MatrixEntry>, DenseVector> {

    private DenseVector vector;
    private double alpha;

    public MatrixEntriesMultiplication(DenseVector vector, double alpha) {

        this.vector = vector;
        this.alpha = alpha;
    }

    @Override
    public Iterator<DenseVector> call(Iterator<MatrixEntry> matrixEntryIterator) throws Exception {

        double result[] = new double[vector.size()];

        for(int i = 0 ; i< result.length; i++) {
            result[i] = 0.0;
        }

        MatrixEntry entry;

        while(matrixEntryIterator.hasNext()) {

            entry = matrixEntryIterator.next();

            result[(int)entry.i()] = result[(int)entry.i()] + (this.vector.apply((int)entry.j()) * entry.value() * this.alpha);

        }

        DenseVector resultVector = new DenseVector(result);

        List<DenseVector> resultList = new ArrayList<DenseVector>();

        resultList.add(resultVector);

        return resultList.iterator();
    }

}
