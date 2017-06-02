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

package com.github.jmabuin.blaspark.examples.options;

import org.apache.commons.cli.*;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * @author Jose M. Abuin
 * @brief Class that provides the default program options
 */
public class GeneralOptions {

    private static final Log LOG = LogFactory.getLog(GeneralOptions.class);

    private Options options = null;

    public enum Mode { HELP, DMXV, SMXV, CG}
    public enum MatrixFormat {PAIRLINE};

    private Mode mode;
    private MatrixFormat matrixFormat;

    private String correctUse =
            "spark-submit --class com.github.jmabuin.blaspark.blaSpark blaSpark-0.0.1.jar [Options] <Matrix file name> <Vector file name> <Output file name>";// [SparkBWA Options] Input.fastq [Input2.fastq] Output\n";


    // Header to show when the program is not launched correctly
    private String header = "\tblaSpark performs linear algebra operations ...\nAvailable operating modes are:\n";

    // Footer to show when the program is not launched correctly
    private String footer = "\nPlease report issues at josemanuel.abuin@usc.es";

    private long iterationNumber = 0;
    private long numPartitions = 0;
    private String otherOptions[];

    private String inputVectorPath;
    private String inputMatrixPath;
    private String outputVectorPath;

    public GeneralOptions(String[] args) {

        this.options = this.initOptions();

        //Parse the given arguments
        CommandLineParser parser = new BasicParser();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(this.options, args, true);

            //We look for the operation mode
            if (cmd.hasOption('h') || cmd.hasOption("help")) {
                //Case of showing the help
                this.mode = Mode.HELP;
            } else if (cmd.hasOption('d') || cmd.hasOption("dmxv")) {
                // Case of query
                this.mode = Mode.DMXV;
            } else if (cmd.hasOption('s') || cmd.hasOption("smxv")) {
                // Case of build
                this.mode = Mode.SMXV;
            } else if (cmd.hasOption('c') || cmd.hasOption("conjGrad")) {
                // Case of add
                this.mode = Mode.CG;
            } else {
                // Default case. Help
                LOG.warn("[" + this.getClass().getName() + "] :: No operation mode selected. Using help ");
                this.mode = Mode.HELP;
            }

            if(cmd.hasOption('l') || cmd.hasOption("pairLine")) {
                this.matrixFormat = MatrixFormat.PAIRLINE;
            }

            if(cmd.hasOption('i') || cmd.hasOption("iteration")) {

                this.iterationNumber = Long.parseLong(cmd.getOptionValue("iteration"));
            }

            if (cmd.hasOption('p') || cmd.hasOption("partitions")) {
                this.numPartitions = Long.parseLong(cmd.getOptionValue("partitions"));

            }

            // Get and parse the rest of the arguments
            this.otherOptions = cmd.getArgs(); //With this we get the rest of the arguments

            if(this.otherOptions.length != 3) {
                LOG.error("Input and output parameters not specified.");
                this.printHelp();
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

    public Options initOptions() {

        Options privateOptions = new Options();

        OptionGroup general = new OptionGroup();

        // Help
        Option help = new Option("h","help", false,"Shows documentation");
        general.addOption(help);

        privateOptions.addOptionGroup(general);

        // Operations
        OptionGroup operations = new OptionGroup();

        Option dmxv = new Option("d", "dmxv", false, "Performs a distributed dense matrix dot vector operation");
        operations.addOption(dmxv);

        Option smxv = new Option("s", "smxv", false, "Performs a distributed sparse matrix dot vector operation");
        operations.addOption(smxv);

        Option conjGrad = new Option("c", "conjGrad", false, "Solves a system with the conjugate gradient method");
        operations.addOption(conjGrad);

        privateOptions.addOptionGroup(operations);


        // Number of iterations for CG
        Option iteration = new Option("i","iteration", true,"Number of iterations to perform");
        //buildOptions.addOption(sketchlen);
        privateOptions.addOption(iteration);


        // Matrix formats
        OptionGroup matrixFormat = new OptionGroup();
        Option pairLine = new Option("l", "pairLine", false, "The matrix format will be a JavaPairRDD<Long, DenseVector>");

        matrixFormat.addOption(pairLine);
        privateOptions.addOptionGroup(matrixFormat);

        // Partition number
        Option partitions = new Option("p","partitions", true,"Number of partitions to divide the matrix");
        privateOptions.addOption(partitions);


        return privateOptions;
    }

    public void printHelp() {
        //To print the help
        HelpFormatter formatter = new HelpFormatter();
        //formatter.setWidth(500);
        formatter.printHelp( correctUse,header, this.options,footer , true);

    }

    public Mode getMode() {
        return mode;
    }

    public MatrixFormat getMatrixFormat() {
        return matrixFormat;
    }

    public String[] getOtherOptions() {
        return otherOptions;
    }

    public long getIterationNumber() {
        return iterationNumber;
    }

    public long getNumPartitions() {
        return numPartitions;
    }

    public String getInputVectorPath() {
        return inputVectorPath;
    }

    public String getInputMatrixPath() {
        return inputMatrixPath;
    }

    public String getOutputVectorPath() {
        return outputVectorPath;
    }
}
