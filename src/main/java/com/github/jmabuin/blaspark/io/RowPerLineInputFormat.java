package com.github.jmabuin.blaspark.io;


import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;


/**
 * This class define an InputFormat for Matrix Market files for the
 * Hadoop MapReduce framework.
 *
 * @author José M. Abuín
 */
public class RowPerLineInputFormat extends FileInputFormat<Long,double[]> {

	@Override
	public RecordReader<Long, double[]> createRecordReader( InputSplit inputSplit, TaskAttemptContext taskAttemptContext) {
		return new RowPerLineRecordReader();
	}


	/*
	@Override
	public boolean isSplitable(JobContext context, Path file) {
		return false;
	}
	*/

}