#!/bin/bash

hdfs dfs -mkdir cifar10
hdfs dfs -put examples/speedo/*_datumfile examples/speedo/mean.binaryproto examples/speedo/*.prototxt cifar10
