# BLASpark
**BLASpark** objective is to perform distributed linear algebra operations using Apache Spark. To do so, **BLASpark** uses some of the inner types provided by Spark, such as _DistributedMatrix_ or _DenseVector_. Typical BLAS L1, L2 and L3 operations are not implemented in a distributed way with Spark and with these data types. With **BLASpark** we will try to give access to the most common BLAS operations in a distributed way with Spark.

## Available operations
So far, the implemented operations in **BLASpark** are:

### L1 BLAS

### L2 BLAS

 - DGEMV
 - DGER

### L3 BLAS

 - DGEMM

## Building BLASpark
**BLASpark** is implemented in Java, and Maven and IntellIJ are used to build and write the code respectively. The build should be performed in the Spark cluster where **BLASpark** will be used. The cluster characteristics should be:
                                                                                                               
 - Java JDK >= 8
 - Hadoop >= 2.7.1 (with YARN and HDFS)
 - Maven >= 3
 - Spark >= 2.0

To build **BLASpark**:                                                                                                                

    git clone https://github.com/jmabuin/BLASpark.git
    cd BLASpark
    mvn clean package

## Using and testing BLASpark
*BLASpark** can be added to a project as a .jar dependency. However, if a user wants to test it, it comes with some examples:

 - Dense matrix dot vector example.
 - Dense matrix dot dense matrix operation.
 - Solve a system by using the Conjugate Gradient method.
 - Solve a system by using the Jacobi method.

The input data for these examples can be generated with the [**Matrix Market Suite**](https://github.com/jmabuin/matrix-market-suite) software. **BLASpark** can read and write matrices from/to files in HDFS where each file line is a row from the matrix. These kind of files can be generated for input data with [**Matrix Market Suite**](https://github.com/jmabuin/matrix-market-suite). As an example, we will perform a matrix dot vector operation with **BLASpark**. For that, we will have two files, _Matrix-16.mtx_ and _Vector-16.mtx_, representing a 16x16 matrix and a 16 item vector.

To launch the example, after building **BLASpark** we will enter into the _target_ directory and run the example as follows:

    cd target
    spark-submit --class com.github.jmabuin.blaspark.examples.BLASpark --master yarn-cluster --driver-memory 1500m --executor-memory 3g --executor-cores 1 --verbose --num-executors 4 ./BLASpark-0.0.1.jar -d -p 4 Matrices/Matriz-16.mtx Matrices/Vector-16.mtx Matrices/Output-16.mtx