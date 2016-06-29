import os,cv2,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from joblib import Parallel, delayed


import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF-8')

#os.environ['SPARK_HOME']="/Users/kyoyachuan/Downloads/spark-1.6.0/" 
#sys.path.append("/Users/kyoyachuan/Downloads/spark-1.6.0/bin") 
#sys.path.append("/Users/kyoyachuan/Downloads/spark-1.6.0/python")
#sys.path.append("/Users/kyoyachuan/Downloads/spark-1.6.0/python/pyspark/")
#sys.path.append("/Users/kyoyachuan/Downloads/spark-1.6.0/python/lib") 
#sys.path.append("/Users/kyoyachuan/Downloads/spark-1.6.0/python/lib/pyspark.zip")
#exec(open('/Users/kyoyachuan/Downloads/spark-1.6.0/python/pyspark/shell.py').read())

import pyspark
from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext

import pyspark.mllib
import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.linalg import SparseVector

from pyspark.mllib.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier

sc = SparkContext('local[*]')
sqlContext = SQLContext(sc)
print("hello")

img_rows, img_cols = 48,64

def get_image(path, img_rows, img_cols, indexed):
    # Load as grayscale

    img = cv2.imread(path, 0)
    #elif color_type == 3:
     #   img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    img = cv2.Canny(resized, 90, 250)
    dims = np.shape(img)
    img_data= np.reshape(img, (1,dims[0] * dims[1]))[0]
    img_data = img_data.astype('float') / 255.
    num_of_features = len(img_data)
    indices_of_sparsedata = [i for i, x in enumerate(img_data) if x != 0]
    values_of_sparsevector = [img_data[i] for i in indices_of_sparsedata]
    inputdata = LabeledPoint(float(indexed), SparseVector(num_of_features, indices_of_sparsedata, values_of_sparsevector))
            
    return inputdata

def load_train(img_rows, img_cols, color_type=1):
    Xtrain = []
    starttime = datetime.datetime.now()
    print('Read train images: %s\n' %(starttime))
    for j in range(10):
        print('Load folder c{}...'.format(j))
        path = os.path.join('/Users/kyoyachuan/Downloads/imgs/', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        #i = 0
        
            #i += 1
            #if i >= 100:
        Xtrain.extend(Parallel(n_jobs=2)(delayed(get_image)(im_file,img_rows,img_cols,j) for im_file in files))           
            
    
    print("\nDone!\nImage Load Processed in %s" % (datetime.datetime.now() - starttime))        
    print("\nTotal images loaded: %s" % len(Xtrain))
    return Xtrain

Xtrain = load_train(img_rows,img_cols)

data = sc.parallelize(Xtrain)

print("\nSplitting data into 60","%"," training and 40","%","testing")
training_data, testing_data = data.randomSplit([0.6, 0.4], seed=0)
vectorizedData = training_data.toDF()
print("Creating MultilayerPerceptronClassifier...")
MLP = MultilayerPerceptronClassifier(labelCol='indexedLabel', featuresCol='indexedFeatures')
labelIndexer = StringIndexer(inputCol='label',
                             outputCol='indexedLabel').fit(vectorizedData)
featureIndexer = VectorIndexer(inputCol='features',
                               outputCol='indexedFeatures',
                               maxCategories=2).fit(data.toDF())
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, MLP])

paramGrid_MLP = ParamGridBuilder().addGrid(MLP.layers,[[3072, neuron, 10] for neuron in [200, 500]]).build()
evaluator = MulticlassClassificationEvaluator(labelCol='indexedLabel',
                                      predictionCol='prediction', metricName='f1')
print("Processing crossvalidation with 3-fold & 200/500 hidden layer units")
crossval = CrossValidator(estimator=pipeline,
                  estimatorParamMaps=paramGrid_MLP,
                  evaluator=evaluator,
                  numFolds=3)
starttime = datetime.datetime.now()
CV_model = crossval.fit(vectorizedData)
print CV_model.bestModel.stages[2]
print('Done on fitting model:%s'%(datetime.datetime.now()-starttime))

print("Transforming testing data...")
vectorized_test_data = testing_data.toDF()

#transformed_data1 = CV_model.transform(vectorizedData)
#print evaluator.getMetricName(), 'accuracy:', evaluator.evaluate(transformed_data1)
transformed_data = CV_model.transform(vectorized_test_data)
#print transformed_data.first()
print("Fitting testing data into model...")
print evaluator.getMetricName(), 'accuracy:', evaluator.evaluate(transformed_data)

predictions = transformed_data.select('indexedLabel', 'prediction')
print predictions.describe().show()
print predictions.take(10)
print predictions.where(predictions.prediction != predictions.indexedLabel)



#predictAndLabel=valid.map(lambda p: (model.predict(p.features),p.label))
#accuracy = 1.0*predictAndLabel.filter(lambda (x, v): x == v).count()/valid.count()
#accuracy
