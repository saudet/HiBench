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
//package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
// $example off$
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

import org.bytedeco.frovedis.frovedis_server

object DecisionTreeRegressionExample {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DecisionTreeRegressionExample")
    val sc = new SparkContext(conf)

    frovedis_server.initialize("-np 8")

    val cacheStart = System.currentTimeMillis()

    // $example on$
    // Load and parse the data file.
    val data: RDD[LabeledPoint] = sc.objectFile(args(0))
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val numExamples = data.count()

    println(s"Loading data time (ms) = ${System.currentTimeMillis() - cacheStart}")
    println(s"numExamples = $numExamples.")

    val trainingStart = System.currentTimeMillis()

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity,
      maxDepth, maxBins)

    println(s"Training time (ms) = ${System.currentTimeMillis() - trainingStart}")

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.collect().map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testSE = labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }
    val testMSE = testSE.sum / testSE.size
    println(s"Test Mean Squared Error = $testMSE")
    println(s"Learned regression tree model:\n ${model.toDebugString}")

    // Save and load model
//    model.save(sc, "/tmp/DecisionTreeRegressionModel")
//    val sameModel = DecisionTreeModel.load(sc, "/tmp/DecisionTreeRegressionModel")
    // $example off$

    frovedis_server.shut_down()
    sc.stop()
  }
}
// scalastyle:on println
