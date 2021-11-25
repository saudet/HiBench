/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package com.intel.hibench.sparkbench.ml
//package org.apache.spark.examples.ml

// $example on$
import org.apache.spark.ml.classification.LinearSVC
// $example off$
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

import org.bytedeco.frovedis.frovedis_server

object LinearSVCExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearSVCExample")
      .getOrCreate()
    val sc = spark.sparkContext

    frovedis_server.initialize("-np 8")

    import spark.implicits._

    val cacheStart = System.currentTimeMillis()

    // $example on$
    // Load training data
    val training = sc.objectFile(args(0)).asInstanceOf[RDD[LabeledPoint]]
      .map(p => (p.label, p.features.asML)).toDF("label", "features")

    val numExamples = training.count()

    println(s"Loading data time (ms) = ${System.currentTimeMillis() - cacheStart}")
    println(s"numExamples = $numExamples.")

    val trainingStart = System.currentTimeMillis()

    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)

    // Fit the model
    val lsvcModel = lsvc.fit(training)

    println(s"Training time (ms) = ${System.currentTimeMillis() - trainingStart}")

    // Print the coefficients and intercept for linear svc
    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
    // $example off$

    frovedis_server.shut_down()
    spark.stop()
  }
}
// scalastyle:on println
