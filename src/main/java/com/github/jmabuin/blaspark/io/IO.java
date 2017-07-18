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

package com.github.jmabuin.blaspark.io;

import com.github.jmabuin.blaspark.operations.L3;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.*;

import java.io.*;
import java.util.List;

/**
 * @author Jose M. Abuin
 * @brief Input/output static methods
 */
public class IO {

	private static final Log LOG = LogFactory.getLog(IO.class);

	public static DenseVector readVectorFromFileInHDFS(String file, Configuration conf){

		try {
			FileSystem fs = FileSystem.get(conf);

			Path pt = new Path(file);

			//FileSystem fileSystem = FileSystem.get(context.getConfiguration());
			BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
			String line;
			line=br.readLine();

			double vector[] = null;

			boolean arrayInfo = true;

			int i = 0;

			while (line != null){

				if((arrayInfo == true) && (line.charAt(0) == '%')){
					arrayInfo = true;
					//LOG.info("JMAbuin:: Skipping line with %");
				}
				else if((arrayInfo == true) && !(line.charAt(0) == '%')){
					arrayInfo = false;
					String[] matrixInfo = line.split(" ");
					//LOG.info("JMAbuin:: Creating vector after line with %");
					vector = new double[Integer.parseInt(matrixInfo[0])];

				}
				else{
					vector[i] = Double.parseDouble(line);
					i++;
				}

				line=br.readLine();
			}

			br.close();

			return new DenseVector(vector);

		} catch (IOException e) {
			LOG.error("Error in " + IO.class.getName() + ": " + e.getMessage());
			e.printStackTrace();
			System.exit(1);
		}

		return null;
	}

	public static void writeVectorToFileInHDFS(String file, DenseVector vector, Configuration conf){

		try {
			FileSystem fs = FileSystem.get(conf);

			Path pt = new Path(file);

			//FileSystem fileSystem = FileSystem.get(context.getConfiguration());
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fs.create(pt, true)));

			bw.write("%%MatrixMarket matrix array real general");
			bw.newLine();
			bw.write(vector.size()+" 1");
			bw.newLine();

			for(int i = 0; i< vector.size(); i++) {
				bw.write(String.valueOf(vector.apply(i)));
				bw.newLine();
			}



			bw.close();
			//fs.close();


		} catch (IOException e) {
			LOG.error("Error in " + IO.class.getName() + ": " + e.getMessage());
			e.printStackTrace();
			System.exit(1);
		}

	}

	public static void writeMatrixToFileInHDFS(String file, DistributedMatrix matrix, Configuration conf){

		try {
			List<IndexedRow> localRows;
			long numRows = 0;
			long numCols = 0;

			FileSystem fs = FileSystem.get(conf);

			Path pt = new Path(file);

			//FileSystem fileSystem = FileSystem.get(context.getConfiguration());
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fs.create(pt, true)));

			JavaRDD<IndexedRow> rows;

			if( matrix.getClass() == IndexedRowMatrix.class) {
				rows = ((IndexedRowMatrix) matrix).rows().toJavaRDD();
			}
			else if (matrix.getClass() == CoordinateMatrix.class) {
				rows = ((CoordinateMatrix)matrix).toIndexedRowMatrix().rows().toJavaRDD();
			}
			else if (matrix.getClass() == BlockMatrix.class){
				rows = ((BlockMatrix)matrix).toIndexedRowMatrix().rows().toJavaRDD();
			}
			else {
				rows = null;
			}

			localRows = rows.collect();

			Vector vectors[] = new Vector[localRows.size()];

			for(int i = 0; i< localRows.size(); i++) {
				vectors[(int)localRows.get(i).index()] = localRows.get(i).vector();
			}

			numRows = matrix.numRows();
			numCols = matrix.numCols();

			bw.write("%%MatrixMarket matrix array real general");
			bw.newLine();
			bw.write(numRows+" "+numCols+" "+(numRows * numCols));
			bw.newLine();

			for(int i = 0; i< vectors.length; i++) {
				bw.write(i+":");
				for(int j = 0; j< vectors[i].size(); j++) {
					bw.write(String.valueOf(vectors[i].apply(j))+",");
				}

				bw.newLine();
			}

			bw.close();
			//fs.close();


		} catch (IOException e) {
			LOG.error("Error in " + IO.class.getName() + ": " + e.getMessage());
			e.printStackTrace();
			System.exit(1);
		}

	}

}
