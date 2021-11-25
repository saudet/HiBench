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

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
// $example on$
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
// $example off$

import org.apache.spark.rdd.RDD
import org.bytedeco.frovedis.frovedis_server

object Word2VecExample {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Word2VecExample")
    val sc = new SparkContext(conf)

    frovedis_server.initialize("-np 8")

    val cacheStart = System.currentTimeMillis()

    // $example on$
    val corpus: RDD[(Long, Vector)] = sc.objectFile(args(0))
    val input = corpus.map { case (id, vec) => vec.toArray.mkString(" ").split(" ").toSeq }

    val numExamples = input.count()

    println(s"Loading data time (ms) = ${System.currentTimeMillis() - cacheStart}")
    println(s"numExamples = $numExamples.")

    val trainingStart = System.currentTimeMillis()

    val word2vec = new Word2Vec()

    val model = word2vec.fit(input)

    println(s"Training time (ms) = ${System.currentTimeMillis() - trainingStart}")

//    val synonyms = model.findSynonyms("1", 5)
//
//    for((synonym, cosineSimilarity) <- synonyms) {
//      println(s"$synonym $cosineSimilarity")
//    }

    // Save and load model
    model.save(sc, "/tmp/Word2VecModel")
    val sameModel = Word2VecModel.load(sc, "/tmp/Word2VecModel")
    // $example off$

    frovedis_server.shut_down()
    sc.stop()
  }
}
// scalastyle:on println
