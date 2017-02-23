package com.github.jmabuin.blaspark.examples;

import org.apache.commons.cli.*;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Created by chema on 2/16/17.
 */
public class ConjugateGradientOptions {

	private static final Log LOG = LogFactory.getLog(ConjugateGradientOptions.class);

	private Options opts;
	private long iterationNumber = 0;
	private long numPartitions = 0;
	private String otherOptions[];

	private String inputVectorPath;
	private String inputMatrixPath;
	private String outputVectorPath;

	public ConjugateGradientOptions(String args[]) {

		this.opts = this.initOptions();

		//Parse the given arguments
		CommandLineParser parser = new BasicParser();
		CommandLine cmd;


		try {
			cmd = parser.parse(this.opts, args);

			//We check the options

			if (cmd.hasOption('i') || cmd.hasOption("iteration")) {
				//Case of sketchlen
				this.iterationNumber = Long.parseLong(cmd.getOptionValue("iteration"));

			}

			if (cmd.hasOption('p') || cmd.hasOption("partitions")) {
				// Case of winlen
				this.numPartitions = Long.parseLong(cmd.getOptionValue("partitions"));

			}


			// Get and parse the rest of the arguments
			this.otherOptions = cmd.getArgs(); //With this we get the rest of the arguments

			// Check if the number of arguments is correct. This is, matrix path, vector path, and solution path
			if (this.otherOptions.length != 3) {
				LOG.error("["+this.getClass().getName()+"] No input matrix, input vector and output vector have been found. Aborting.");

				for (String tmpString : this.otherOptions) {
					LOG.error("["+this.getClass().getName()+"] Other args:: " + tmpString);
				}

				//formatter.printHelp(correctUse, header, options, footer, true);
				System.exit(1);
			}
			else {

				this.inputMatrixPath = this.otherOptions[0];
				this.inputVectorPath = this.otherOptions[1];
				this.outputVectorPath = this.otherOptions[2];

			}
		}
		catch (UnrecognizedOptionException e) {
			e.printStackTrace();
			//formatter.printHelp(correctUse, header, options, footer, true);

			System.exit(1);
		} catch (MissingOptionException e) {
			//formatter.printHelp(correctUse, header, options, footer, true);
			System.exit(1);
		} catch (ParseException e) {
			//formatter.printHelp( correctUse,header, options,footer , true);
			e.printStackTrace();
			System.exit(1);
		}

	}

	public Options getOpts() {
		return opts;
	}

	public void setOpts(Options opts) {
		this.opts = opts;
	}

	public long getIterationNumber() {
		return iterationNumber;
	}

	public void setIterationNumber(long iterationNumber) {
		this.iterationNumber = iterationNumber;
	}

	public long getNumPartitions() {
		return numPartitions;
	}

	public void setNumPartitions(long numPartitions) {
		this.numPartitions = numPartitions;
	}

	public String getInputVectorPath() {
		return inputVectorPath;
	}

	public void setInputVectorPath(String inputVectorPath) {
		this.inputVectorPath = inputVectorPath;
	}

	public String getInputMatrixPath() {
		return inputMatrixPath;
	}

	public void setInputMatrixPath(String inputMatrixPath) {
		this.inputMatrixPath = inputMatrixPath;
	}

	public String getOutputVectorPath() {
		return outputVectorPath;
	}

	public void setOutputVectorPath(String outputVectorPath) {
		this.outputVectorPath = outputVectorPath;
	}

	public Options initOptions() {

		Options opt = new Options();

		Option iteration = new Option("i","iteration", true,"Number of iterations to perform");
		//buildOptions.addOption(sketchlen);
		opt.addOption(iteration);

		Option partitions = new Option("p","partitions", true,"Number of partitions to divide the matrix");
		opt.addOption(partitions);

		return opt;
	}

}
