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
import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.linalg.distributed.*;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by chema on 7/12/17.
 */
public class L3 {

    private static final Log LOG = LogFactory.getLog(L3.class);

    /* ****************************************** DGEMM ****************************************** */
    // C := alpha*op( A )*op( B ) + beta*C
    public static DistributedMatrix DGEMM(double alpha, DistributedMatrix A, DistributedMatrix B, double beta, DistributedMatrix C, JavaSparkContext jsc){

        if (beta != 1.0) {
            if (beta != 0.0) {
                C = L3.SCAL(beta, C, C, jsc);
            }
        }

        if (alpha == 0.0) {
            return C;
        }

        else if(alpha == 1.0){
            if( A.getClass() == IndexedRowMatrix.class) {
                C = L3.DGEMM_IRW((IndexedRowMatrix)A, (IndexedRowMatrix) B, jsc);
            }
            else if (A.getClass() == CoordinateMatrix.class) {
                C = L3.DGEMM_COORD((CoordinateMatrix)A, (CoordinateMatrix) B, jsc);
            }
            else if (A.getClass() == BlockMatrix.class){
                C = L3.DGEMM_BCK((BlockMatrix)A, (BlockMatrix) B, jsc);
            }
            else {
                C = null;
            }
        }
        else {
            if( A.getClass() == IndexedRowMatrix.class) {
                C = L3.DGEMM_IRW((IndexedRowMatrix)A, (IndexedRowMatrix) B, jsc);
                C = L3.SCAL(alpha, C, C, jsc);
            }
            else if (A.getClass() == CoordinateMatrix.class) {
                C = L3.DGEMM_COORD((CoordinateMatrix)A, (CoordinateMatrix) B, jsc);
                C = L3.SCAL(alpha, C, C, jsc);
            }
            else if (A.getClass() == BlockMatrix.class){
                C = L3.DGEMM_BCK((BlockMatrix)A, (BlockMatrix) B, jsc);
                C = L3.SCAL(alpha, C, C, jsc);
            }
            else {
                C = null;
            }
        }


        return C;
    }

    // B := alpha * A
    public static DistributedMatrix SCAL(double alpha, DistributedMatrix A, DistributedMatrix B, JavaSparkContext jsc){

        if( A.getClass() == IndexedRowMatrix.class) {
            B = L3.SCAL_IRW(alpha, (IndexedRowMatrix) A, (IndexedRowMatrix) B, jsc);
        }
        else if (A.getClass() == CoordinateMatrix.class) {
            B = L3.SCAL_COORD(alpha, (CoordinateMatrix) A, (CoordinateMatrix) B, jsc);
        }
        else if (A.getClass() == BlockMatrix.class){
            B = L3.SCAL_BCK(alpha, (BlockMatrix) A, (BlockMatrix) B, jsc);
        }
        else {
            B = null;
        }

        return B;

    }

    private static IndexedRowMatrix SCAL_IRW(double alpha, IndexedRowMatrix A, IndexedRowMatrix B, JavaSparkContext jsc) {

        JavaRDD<IndexedRow> rows = A.rows().toJavaRDD();

        final Broadcast<Double> alphaBC = jsc.broadcast(alpha);

        JavaRDD<IndexedRow> newRows = rows.map(new Function<IndexedRow, IndexedRow>() {
            @Override
            public IndexedRow call(IndexedRow indexedRow) throws Exception {

                double alphaValue = alphaBC.getValue().doubleValue();

                long index = indexedRow.index();

                double values[] = new double[indexedRow.vector().size()];

                for(int i = 0; i< values.length; i++) {
                    values[i] = indexedRow.vector().apply(i) * alphaValue;
                }

                return new IndexedRow(index, new DenseVector(values));

            }
        });

        B = new IndexedRowMatrix(newRows.rdd());

        return B;

    }

    private static CoordinateMatrix SCAL_COORD(double alpha, CoordinateMatrix A, CoordinateMatrix B, JavaSparkContext jsc) {

        JavaRDD<MatrixEntry> entries = A.entries().toJavaRDD();

        final Broadcast<Double> alphaBC = jsc.broadcast(alpha);

        JavaRDD<MatrixEntry> newEntries = entries.mapPartitions(new FlatMapFunction<Iterator<MatrixEntry>, MatrixEntry>() {
            @Override
            public Iterator<MatrixEntry> call(Iterator<MatrixEntry> matrixEntryIterator) throws Exception {

                double alphaValue = alphaBC.getValue().doubleValue();

                List<MatrixEntry> newEntries = new ArrayList<MatrixEntry>();

                while(matrixEntryIterator.hasNext()) {
                    MatrixEntry currentEntry = matrixEntryIterator.next();

                    newEntries.add(new MatrixEntry(currentEntry.i(), currentEntry.j(), currentEntry.value() * alphaValue));

                }

                return newEntries.iterator();
            }
        });

        B = new CoordinateMatrix(newEntries.rdd());

        return B;

    }


    private static BlockMatrix SCAL_BCK(double alpha, BlockMatrix A, BlockMatrix B, JavaSparkContext jsc) {

        JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> blocks = A.blocks().toJavaRDD();

        final Broadcast<Double> alphaBC = jsc.broadcast(alpha);

        JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> newBlocks = blocks.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {
            @Override
            public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> block) throws Exception {

                double alphaBCRec = alphaBC.getValue().doubleValue();

                Integer row = (Integer)block._1._1; //Integer.parseInt(block._1._1.toString());
                Integer col = (Integer)block._1._2;
                Matrix matrixBlock = block._2;

                for(int i = 0; i< matrixBlock.numRows(); i++) {

                    for(int j = 0; j< matrixBlock.numCols(); j++) {
                        matrixBlock.update(i,j, matrixBlock.apply(i,j) * alphaBCRec);
                    }

                }

                return new Tuple2<Tuple2<Object, Object>, Matrix>(new Tuple2<Object, Object>(row, col), matrixBlock);

            }
        });

        B = new BlockMatrix(newBlocks.rdd(), A.rowsPerBlock(), A.colsPerBlock());

        return B;

    }

    public static IndexedRowMatrix DGEMM_IRW( IndexedRowMatrix A, IndexedRowMatrix B, JavaSparkContext jsc){

        /*IndexedRow[] rowsB = B.rows().collect();

        final Broadcast<IndexedRow[]> rowsBC = jsc.broadcast(rowsB);

        JavaRDD<IndexedRow> newRows = A.rows().toJavaRDD().map(new Function<IndexedRow, IndexedRow>() {
            @Override
            public IndexedRow call(IndexedRow indexedRow) throws Exception {

                IndexedRow[] rowsBValues = rowsBC.getValue();

                double newRowValues[] = new double[rowsBValues[0].vector().size()];

                for(int i = 0; i< newRowValues.length ; i++) {
                    newRowValues[i] = 0.0;
                }

                Vector vect = indexedRow.vector();

                for(int i = 0; i< rowsBValues.length; i++) {

                    newRowValues[i] = BLAS.dot(vect, rowsBValues[i].vector());

                }

                Vector result = new DenseVector(newRowValues);

                return new IndexedRow(indexedRow.index(), result);

            }
        });

        return new IndexedRowMatrix(newRows.rdd());*/

        Matrix matrixB = L3.toLocal(B);

        return A.multiply(matrixB);


    }


    public static CoordinateMatrix DGEMM_COORD( CoordinateMatrix A, CoordinateMatrix B, JavaSparkContext jsc){

        // Temporal method
        return DGEMM_IRW(A.toIndexedRowMatrix(), B.toIndexedRowMatrix(), jsc).toCoordinateMatrix();


    }

    public static BlockMatrix DGEMM_BCK(BlockMatrix A, BlockMatrix B, JavaSparkContext jsc) {

        return DGEMM_IRW(A.toIndexedRowMatrix(), B.toIndexedRowMatrix(), jsc).toBlockMatrix();
    }


    private static Matrix toLocal(IndexedRowMatrix A) {

        int numRows = (int)A.numRows();
        int numCols = (int)A.numCols();
        List<IndexedRow> rows = A.rows().toJavaRDD().collect();

        Vector vectors[] = new Vector[rows.size()];

        for(int i = 0; i< rows.size(); i++) {
            vectors[(int)rows.get(i).index()] = rows.get(i).vector();
        }


        double values[] = new double[numRows * numCols];

        for(int i = 0; i< numRows; i++) {

            for(int j = 0; j < numCols; j++) {

                values[j * numRows + i] = vectors[i].apply(j);

            }

        }

        return new DenseMatrix(numRows, numCols, values);


    }

    private static Matrix toLocal(CoordinateMatrix A) {

        int numRows = (int)A.numRows();
        int numCols = (int)A.numCols();
        List<MatrixEntry> entries = A.entries().toJavaRDD().collect();

        double values[] = new double[numRows * numCols];

        for(MatrixEntry currentEntry: entries) {
            values[(int)(currentEntry.j() * numRows + currentEntry.i())] = currentEntry.value();
        }

        return new DenseMatrix(numRows, numCols, values);


    }

}
