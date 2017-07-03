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
    public enum MatrixFormat {PAIRLINE, COORDINATE, BLOCK};

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

    private int rowsPerlBlock = 0;
    private int colsPerBlock = 0;

    private double alpha;
    private double beta;

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
                LOG.warn("[" + this.getClass().getName() + "] :: No operation mode selected. Using help.");
                this.mode = Mode.HELP;
            }

            if(cmd.hasOption('l') || cmd.hasOption("pairLine")) {
                this.matrixFormat = MatrixFormat.PAIRLINE;
            }
            else if(cmd.hasOption('o') || cmd.hasOption("coordinate")) {
                this.matrixFormat = MatrixFormat.COORDINATE;
            }
            else if(cmd.hasOption('b') || cmd.hasOption("blocked")) {
                this.matrixFormat = MatrixFormat.BLOCK;

                if(!cmd.hasOption("rows") || !cmd.hasOption("cols")) {
                    LOG.error("[" + this.getClass().getName() + "] :: The number of rows or cols has not been specified.");
                    this.printHelp();
                    System.exit(1);
                }
                else {
                    this.colsPerBlock = Integer.parseInt(cmd.getOptionValue("cols"));
                    this.rowsPerlBlock = Integer.parseInt(cmd.getOptionValue("rows"));
                }

            }
            else {
                this.matrixFormat = MatrixFormat.PAIRLINE;
            }

            if(cmd.hasOption('i') || cmd.hasOption("iteration")) {

                this.iterationNumber = Long.parseLong(cmd.getOptionValue("iteration"));
            }

            if (cmd.hasOption('p') || cmd.hasOption("partitions")) {
                this.numPartitions = Long.parseLong(cmd.getOptionValue("partitions"));

            }

            if(cmd.hasOption("alpha")) {
                this.alpha = Double.parseDouble(cmd.getOptionValue("alpha"));
            }
            else {
                this.alpha = 1.0;
            }

            if(cmd.hasOption("beta")) {
                this.beta = Double.parseDouble(cmd.getOptionValue("beta"));
            }
            else {
                this.beta = 0.0;
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

        // Options: h, d, s, c, i, l, o, b, p

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
        privateOptions.addOption(iteration);


        // Matrix formats
        OptionGroup matrixFormat = new OptionGroup();
        Option pairLine = new Option("l", "pairLine", false, "The matrix format will be a IndexedRowMatrix");
        matrixFormat.addOption(pairLine);

        Option coordinate = new Option("o", "coordinate", false, "The matrix format will be a CoordinateMatrix");
        matrixFormat.addOption(coordinate);

        Option blocked = new Option("b", "blocked", false, "The matrix format will be a BlockMatrix");
        matrixFormat.addOption(blocked);

        privateOptions.addOptionGroup(matrixFormat);

        // Partition number
        Option partitions = new Option("p","partitions", true,"Number of partitions to divide the matrix");
        privateOptions.addOption(partitions);

        // Rows and cols per block for blocked format
        Option rowsPerBlock = new Option(null,"rows", true,"Number of rows for block in BlockMatrix format");
        privateOptions.addOption(rowsPerBlock);

        Option colsPerBlock = new Option(null,"cols", true,"Number of cols for block in BlockMatrix format");
        privateOptions.addOption(colsPerBlock);

        // Alpha and beta for DMxV operation
        Option alpha = new Option(null, "alpha", true, "Alpha value for DMxV example");
        privateOptions.addOption(alpha);

        Option beta = new Option(null, "beta", true, "Beta value for DMxV example");
        privateOptions.addOption(beta);

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

    public int getRowsPerlBlock() {
        return rowsPerlBlock;
    }

    public int getColsPerBlock() {
        return colsPerBlock;
    }

    public double getAlpha() {
        return alpha;
    }

    public double getBeta() {
        return beta;
    }
}
