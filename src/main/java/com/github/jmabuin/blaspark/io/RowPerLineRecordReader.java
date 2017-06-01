package com.github.jmabuin.blaspark.io;

import java.io.IOException;

import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;

/**
 * This class define a custom RecordReader for RowPerLine files for the
 * Hadoop MapReduce framework.

 * @author José M. Abuín
 */
public class RowPerLineRecordReader extends RecordReader<Long, double[]> {

	// input data comes from lrr
	private LineRecordReader lrr = null;

	private Long key = 0L;
	private double[] value;

	private boolean firstSplit = false;


	@Override
	public void close() throws IOException {
		this.lrr.close();
	}

	@Override
	public Long getCurrentKey()
			throws IOException, InterruptedException {
		return key;
	}

	@Override
	public double[] getCurrentValue()
			throws IOException, InterruptedException {
		return value;
	}

	@Override
	public float getProgress()
			throws IOException, InterruptedException {
		return this.lrr.getProgress();
	}

	@Override
	public void initialize(final InputSplit inputSplit,
						   final TaskAttemptContext taskAttemptContext)
			throws IOException, InterruptedException {
		this.lrr = new LineRecordReader();
		this.lrr.initialize(inputSplit, taskAttemptContext);

	}

	@Override
	public boolean nextKeyValue()
			throws IOException, InterruptedException {
		boolean found = false;


		while (!found) {

			if (!this.lrr.nextKeyValue()) {
				return false;
			}

			final String s = this.lrr.getCurrentValue().toString().trim();


			//System.out.println("nextKeyValue() s="+s);

			// Prevent empty lines
			if (s.length() == 0) {
				continue;
			}

			if (s.charAt(0) == '%'){
				this.firstSplit = true;
				//this.firstValues = true;
			}
			else{
				//To skip the first values in the first split, that are the matrix row number, col number and not zero values
				if(!this.firstSplit){
					String[] dataLine = s.split(":");

					if(dataLine.length!=2){
						return false;
					}

					this.key = Long.parseLong(dataLine[0]);

					String doubleValuesString[] = dataLine[1].split(",");

					double[] doubleValues = new double[doubleValuesString.length];


					int i = 0;

					for(String newValue: doubleValuesString){
						doubleValues[i] = Double.parseDouble(newValue);

						i++;
					}


					this.value = doubleValues.clone();
					found = true;
				}
				else{
					this.firstSplit = false;
				}
			}



		} //end-while



		return true;
	}


}