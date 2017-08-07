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

