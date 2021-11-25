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

import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.sql.SparkSession

import org.bytedeco.frovedis.frovedis_server

/**
 * Example for mining frequent itemsets using FP-growth.
 */
object FPGrowthExample {

  case class Params(
    input: String = null,
    minSupport: Double = 0.3,
    numPartition: Int = 2) extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("FPGrowthExample") {
      head("FPGrowth: an example FP-growth app.")
      opt[Double]("minSupport")
        .text(s"minimal support level, default: ${defaultParams.minSupport}")
        .action((x, c) => c.copy(minSupport = x))
      opt[Int]("numPartition")
        .text(s"number of partition, default: ${defaultParams.numPartition}")
        .action((x, c) => c.copy(numPartition = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"FPGrowthExample with $params")
      .getOrCreate()
    val sc = spark.sparkContext

    frovedis_server.initialize("-np 8")

    val cacheStart = System.currentTimeMillis()

    val data = Array.fill[String](1000) {
      var s: String = ""
      var letters = (0 until 26).map('a' + _).to[scala.collection.mutable.ListBuffer]
      for (n <- 0 to scala.util.Random.nextInt(26)) {
         s += letters.remove(scala.util.Random.nextInt(letters.length)) + " "
      }
      s
    }
    val transactions = sc.parallelize(data.toSeq).map(t => t.split(" ")).cache()

    val numExamples = transactions.count()

    println(s"Loading data time (ms) = ${System.currentTimeMillis() - cacheStart}")
    println(s"numExamples = $numExamples.")
//    println(s"Number of transactions: ${transactions.count()}")

    val trainingStart = System.currentTimeMillis()

    val model = new FPGrowth()
      .setMinSupport(params.minSupport)
      .setNumPartitions(params.numPartition)
      .run(transactions)

    println(s"Training time (ms) = ${System.currentTimeMillis() - trainingStart}")
    println(s"Number of frequent itemsets: ${model.freqItemsets.collect().length}")

    model.freqItemsets.collect().foreach { itemset =>
      println(s"${itemset.items.mkString("[", ",", "]")}, ${itemset.freq}")
    }

    frovedis_server.shut_down()
    sc.stop()
  }
}
// scalastyle:on println
