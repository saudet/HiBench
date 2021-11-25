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
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.regression.{FMRegressionModel, FMRegressor}
// $example off$
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

import org.bytedeco.frovedis.frovedis_server

object FMRegressorExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("FMRegressorExample")
      .getOrCreate()
    val sc = spark.sparkContext

    frovedis_server.initialize("-np 8")

    import spark.implicits._

    val cacheStart = System.currentTimeMillis()

    // $example on$
    // Load and parse the data file, converting it to a DataFrame.
    val data = sc.objectFile(args(0)).asInstanceOf[RDD[LabeledPoint]]
      .map(p => (p.label, p.features.asML)).toDF("label", "features")

    // Scale features.
    val featureScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val numExamples = data.count()

    println(s"Loading data time (ms) = ${System.currentTimeMillis() - cacheStart}")
    println(s"numExamples = $numExamples.")

    val trainingStart = System.currentTimeMillis()

    // Train a FM model.
    val fm = new FMRegressor()
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
      .setStepSize(0.001)

    // Create a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureScaler, fm))

    // Train model.
    val model = pipeline.fit(trainingData)

    println(s"Training time (ms) = ${System.currentTimeMillis() - trainingStart}")

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "label", "scaledFeatures").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    val fmModel = model.stages(1).asInstanceOf[FMRegressionModel]
    println(s"Factors: ${fmModel.factors} Linear: ${fmModel.linear} " +
      s"Intercept: ${fmModel.intercept}")
    // $example off$

    frovedis_server.shut_down()
    spark.stop()
  }
}
// scalastyle:on println
