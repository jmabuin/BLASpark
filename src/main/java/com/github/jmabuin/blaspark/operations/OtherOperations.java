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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by chema on 7/12/17.
 */
public class OtherOperations {

    private static final Log LOG = LogFactory.getLog(OtherOperations.class);

    public static DistributedMatrix[] GetLU(DistributedMatrix A, boolean diagonalInL, boolean diagonalInU, JavaSparkContext jsc) {

        if((diagonalInL && diagonalInU) || (!diagonalInL && !diagonalInU)) {
            LOG.error("Diagonal values must be in either upper or lower matrices");
            System.exit(-1);
        }

        DistributedMatrix[] returnedMatrices;

        if( A.getClass() == IndexedRowMatrix.class) {
            returnedMatrices = OtherOperations.GetLU_IRW((IndexedRowMatrix) A, diagonalInL, diagonalInU, jsc);
        }
        else if (A.getClass() == CoordinateMatrix.class) {
            returnedMatrices = OtherOperations.GetLU_COORD((CoordinateMatrix) A, diagonalInL, diagonalInU, jsc);
        }
        else if (A.getClass() == BlockMatrix.class){
            // TODO: Implement this operation
            //returnedMatrices = OtherOperations.GetLU_BCK((BlockMatrix) A, diagonalInL, diagonalInU, jsc);
            returnedMatrices = null;
        }
        else {
            returnedMatrices = null;
        }


        return returnedMatrices;

    }

    private static IndexedRowMatrix[] GetLU_IRW(IndexedRowMatrix A, boolean diagonalInL, boolean diagonalInU, JavaSparkContext jsc) {

        IndexedRowMatrix[] returnMatrices = new IndexedRowMatrix[2];

        JavaRDD<IndexedRow> rows = A.rows().toJavaRDD().cache();

        final Broadcast<Boolean> diagonalInLBC = jsc.broadcast(diagonalInL);
        final Broadcast<Boolean> diagonalInUBC = jsc.broadcast(diagonalInU);

        JavaRDD<IndexedRow> upperRows = rows.map(new Function<IndexedRow, IndexedRow>() {

            @Override
            public IndexedRow call(IndexedRow indexedRow) throws Exception {

                boolean diagonalInLValue = diagonalInLBC.getValue().booleanValue();
                boolean diagonalInUValue = diagonalInUBC.getValue().booleanValue();


                long index = indexedRow.index();
                DenseVector vect = indexedRow.vector().toDense();

                double newValues[] = new double[vect.size()];

                if(diagonalInLValue) {
                    for(int i = 0; i< vect.size(); i++) {

                        if( i > index) {
                            newValues[i] = vect.apply(i);
                        }
                        else {
                            newValues[i] = 0.0;
                        }


                    }
                }
                else if(diagonalInUValue) {
                    for(int i = 0; i< vect.size(); i++) {

                        if( i >= index) {
                            newValues[i] = vect.apply(i);
                        }
                        else {
                            newValues[i] = 0.0;
                        }

                    }
                }

                DenseVector newVector = new DenseVector(newValues);

                return new IndexedRow(index, newVector);

            }
        });

        IndexedRowMatrix newUpperMatrix = new IndexedRowMatrix(upperRows.rdd());

        JavaRDD<IndexedRow> lowerRows = rows.map(new Function<IndexedRow, IndexedRow>() {

            @Override
            public IndexedRow call(IndexedRow indexedRow) throws Exception {

                boolean diagonalInLValue = diagonalInLBC.getValue().booleanValue();
                boolean diagonalInUValue = diagonalInUBC.getValue().booleanValue();


                long index = indexedRow.index();
                DenseVector vect = indexedRow.vector().toDense();

                double newValues[] = new double[vect.size()];

                if(diagonalInLValue) {
                    for(int i = 0; i< vect.size(); i++) {

                        if( i <= index) {
                            newValues[i] = vect.apply(i);
                        }
                        else {
                            newValues[i] = 0.0;
                        }


                    }
                }
                else if(diagonalInUValue) {
                    for(int i = 0; i< vect.size(); i++) {

                        if( i < index) {
                            newValues[i] = vect.apply(i);
                        }
                        else {
                            newValues[i] = 0.0;
                        }

                    }
                }

                DenseVector newVector = new DenseVector(newValues);

                return new IndexedRow(index, newVector);

            }
        });

        IndexedRowMatrix newLowerMatrix = new IndexedRowMatrix(lowerRows.rdd());


        returnMatrices[0] = newLowerMatrix;
        returnMatrices[1] = newUpperMatrix;

        return returnMatrices;
    }


    private static CoordinateMatrix[] GetLU_COORD(CoordinateMatrix A, boolean diagonalInL, boolean diagonalInU, JavaSparkContext jsc) {

        CoordinateMatrix[] returnMatrices = new CoordinateMatrix[2];

        JavaRDD<MatrixEntry> rows = A.entries().toJavaRDD().cache();

        final Broadcast<Boolean> diagonalInLBC = jsc.broadcast(diagonalInL);
        final Broadcast<Boolean> diagonalInUBC = jsc.broadcast(diagonalInU);

        JavaRDD<MatrixEntry> lowerRows = rows.mapPartitions(new FlatMapFunction<Iterator<MatrixEntry>, MatrixEntry>() {
            @Override
            public Iterator<MatrixEntry> call(Iterator<MatrixEntry> matrixEntryIterator) throws Exception {
                List<MatrixEntry> newLowerEntries = new ArrayList<MatrixEntry>();

                boolean diagonalInLValue = diagonalInLBC.getValue().booleanValue();
                boolean diagonalInUValue = diagonalInUBC.getValue().booleanValue();

                if(diagonalInLValue){
                    while(matrixEntryIterator.hasNext()) {
                        MatrixEntry currentEntry = matrixEntryIterator.next();

                        if(currentEntry.i() <= currentEntry.j()) {
                            newLowerEntries.add(currentEntry);
                        }
                        else {
                            newLowerEntries.add(new MatrixEntry(currentEntry.i(), currentEntry.j(), 0.0));
                        }


                    }
                }
                else if(diagonalInUValue) {
                    while(matrixEntryIterator.hasNext()) {
                        MatrixEntry currentEntry = matrixEntryIterator.next();

                        if(currentEntry.i() < currentEntry.j()) {
                            newLowerEntries.add(currentEntry);
                        }
                        else {
                            newLowerEntries.add(new MatrixEntry(currentEntry.i(), currentEntry.j(), 0.0));
                        }


                    }
                }


                return newLowerEntries.iterator();
            }
        });

        JavaRDD<MatrixEntry> upperRows = rows.mapPartitions(new FlatMapFunction<Iterator<MatrixEntry>, MatrixEntry>() {
            @Override
            public Iterator<MatrixEntry> call(Iterator<MatrixEntry> matrixEntryIterator) throws Exception {
                List<MatrixEntry> newLowerEntries = new ArrayList<MatrixEntry>();

                boolean diagonalInLValue = diagonalInLBC.getValue().booleanValue();
                boolean diagonalInUValue = diagonalInUBC.getValue().booleanValue();

                if(diagonalInLValue){
                    while(matrixEntryIterator.hasNext()) {
                        MatrixEntry currentEntry = matrixEntryIterator.next();

                        if(currentEntry.i() > currentEntry.j()) {
                            newLowerEntries.add(currentEntry);
                        }
                        else {
                            newLowerEntries.add(new MatrixEntry(currentEntry.i(), currentEntry.j(), 0.0));
                        }


                    }
                }
                else if(diagonalInUValue) {
                    while(matrixEntryIterator.hasNext()) {
                        MatrixEntry currentEntry = matrixEntryIterator.next();

                        if(currentEntry.i() >= currentEntry.j()) {
                            newLowerEntries.add(currentEntry);
                        }
                        else {
                            newLowerEntries.add(new MatrixEntry(currentEntry.i(), currentEntry.j(), 0.0));
                        }


                    }
                }


                return newLowerEntries.iterator();
            }
        });

        CoordinateMatrix newUpperMatrix = new CoordinateMatrix(upperRows.rdd());
        CoordinateMatrix newLowerMatrix = new CoordinateMatrix(lowerRows.rdd());

        returnMatrices[0] = newLowerMatrix;
        returnMatrices[1] = newUpperMatrix;

        return returnMatrices;
    }
}
