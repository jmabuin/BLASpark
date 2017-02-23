package com.github.jmabuin.blaspark.io;

import com.github.jmabuin.blaspark.examples.ConjugateGradientExample;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.mllib.linalg.DenseVector;

import java.io.*;

/**
 * Created by chema on 2/16/17.
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
}
