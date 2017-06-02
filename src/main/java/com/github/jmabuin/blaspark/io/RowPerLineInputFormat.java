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