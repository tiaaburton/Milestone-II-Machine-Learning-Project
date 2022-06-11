<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
'''Used for first ML model in our pipeline.'''
=======
>>>>>>> 5b0eb06c41cea5fb7ef35dc5e473ab7dd93028c3
X_train_spark = spark.createDataFrame(X_train)
X_val_spark = spark.createDataFrame(X_val)
y_train_spark = ps.from_pandas(y_train)
y_val_spark = ps.from_pandas(y_val)
<<<<<<< HEAD
=======
import pandas as pd
import numpy as np
import os
import sys
import random
import time
import ast
import pickle

from google.cloud import bigquery
from google.colab import auth, files
from typing import Union
from spark_sklearn import Converter

import pyspark
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml import *
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_classif

import chart_studio.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

def pyspark_column_prep(dataset):
  """Returns column names with underscores for PySpark ready format.
     dataset: pd.DataFrame"""
  for col in dataset.columns:
    if '.' in col:
      repl = col.replace('.','_').replace(' ','_')
      dataset.rename(columns={col: repl},inplace=True)
  return dataset

def downsampling(data, n):
  df = data.withColumn('rand_col', rand())
  balanced_data = df.withColumn(
    "row_num",row_number().over(Window.partitionBy("label").orderBy("rand_col")))\
    .filter(col("row_num")<=n)\
    .drop("rand_col", "row_num")
  return balanced_data

def take_random_sample(data):
  return data.sample(True, .1, RANDOM_SEED)

def get_mutual_info_features(train_df, target_df, thres:float):
  MI = mutual_info_classif(train_df, target_df)
  mi_results = sorted(list(zip(MI, train_df.columns)), key=lambda x: x[0], reverse=True)

  mi_features = []
  for idx,item in enumerate(mi_results):
    val, col = item
    if val > thres:
      mi_features.append(col)
  mi_features.append('label')

  return mi_features,mi_results

def cv_model(model, train, test, param_grid=None):
  if not param_grid: param_grid = ParamGridBuilder().build()
  cols = [col for col in train.columns if col not in set(['totals_transactions','label','fullVisitorId'])]
  vec = VectorAssembler(inputCols=cols, outputCol='features', handleInvalid='skip')
  pipeline = Pipeline(stages=[vec, model])

  evaluator = BinaryClassificationEvaluator()

  crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=param_grid,
                            evaluator=evaluator,
                            numFolds=4) 
  # increased the number of folds from 4 to 10 after initial testing as 10, time increased drastically
  
  crossval = crossval.fit(train)
  best_model = crossval.bestModel.stages[-1]
  validation_res = crossval.transform(train)
  prediction_res = crossval.transform(test)
  
  train_eval = crossval.getEvaluator().evaluate(validation_res)
  test_eval = crossval.getEvaluator().evaluate(prediction_res)
  return best_model, train_eval, test_eval, validation_res, prediction_res

def save_model(model, file_loc, model_name, sklearn_model=True):
  filename = file_loc + '/' + f'a1_{model_name}_model.sav'
  if sklearn_model == True:
    model = convert_model(model)
    pickle.dump(model, open(filename, 'wb'))
  else:
    try:
      model.save(filename)
    except:
      model.write().overwrite().save(filename)
  return

def convert_model(spark_model):
  conv = Converter(spark)
  sklearn_model = conv.toSKLearn(spark_model)
  return sklearn_model

def get_clf_metrics(data:pyspark.sql.dataframe.DataFrame, true:str, pred:str):
  pred_df = data.select(true,pred).toPandas()
  recall = recall_score(pred_df['label'], pred_df['prediction'])
  f1Score = f1_score(pred_df['label'], pred_df['prediction'])
  precision = precision_score(pred_df['label'], pred_df['prediction'])
  return recall, f1Score, precision

def train_spark_models(spark_datasets, spark_models, param_grids, filters):

  trained_models = []
  df = pd.DataFrame()
  datasets = []

  for model in ['spark_lr', 'spark_dt', 'spark_rf']:
    for names, data in spark_datasets.items():
      train_name, test_name = names
      X_train, X_test = data

      for filter in filters:

        if model in spark_models.keys():

          start_time = time.time()
          trained_model, train_eval, test_eval, validation, predictions = \
          cv_model(spark_models[model],X_train[filter], X_test[filter], param_grid=param_grids[model])

          try:
            best_params = trained_model._java_obj.extractParamMap()
          except:
            best_params = trained_model._java_obj.parent().extractParamMap()
          end_time = time.time()

          datasets.append((validation, predictions))
          trained_models.append(trained_model)

          train_recall, train_f1Score, train_precision = get_clf_metrics(validation,'label','prediction')
          test_recall, test_f1Score, test_precision = get_clf_metrics(predictions,'label','prediction')

          df2 = pd.DataFrame([[model, train_name, test_name, filter, len(filter), train_eval, train_recall, 
                               train_precision, train_f1Score, best_params, start_time, end_time]],
                      columns=['Model', 'Train_Dataset', 'Test_Dataset', 'Column_Filter', 'Col_Num',
                               'AUC', 'Recall', 'Precision', 'F1-Score', 'Best_Params',
                               'Start_Time', 'End_Time'])
          df3 = pd.DataFrame([[model, train_name, test_name, filter, len(filter), test_eval, test_recall, 
                               test_precision, test_f1Score, best_params, start_time, end_time]],
                      columns=['Model', 'Train_Dataset', 'Test_Dataset', 'Column_Filter', 'Col_Num',
                               'AUC', 'Recall', 'Precision', 'F1-Score', 'Best_Params',
                               'Start_Time', 'End_Time'])
          if len(df) == 0:
            df = pd.concat([df2, df3], ignore_index=True)
          else:
            df = pd.concat([df, df2, df3], ignore_index=True)

  return trained_models, df, datasets

def post_processing(df):
  for idx,param_grid in enumerate(df.Best_Params):
    # print(param_grid[3:-2].split(',\n\t'))
    for param_str in (param_grid[3:-2].split(',\n\t')):
      param = param_str.split(':')
      param[0] = param[0].split('-')[-1]
      # param[1] = param[1].strip()

      if param[0] not in df.columns:
        df[param[0]] = np.nan
      elif param[0] in df.columns and len(param) > 1: 
        df[param[0]].iloc[idx] = param[1]
  return df

def save_models(models, force:bool=False):

  for idx,model in enumerate(models):
    try:
      col_count = len(df.iloc[idx]['Column_Filter'])
      f_name = df.iloc[idx]['Model'] + f'_{col_count}MIcol_run{idx}' #+ '_' + df.iloc[idx]['datasets'][0].split('_')[-1]
      path_name = f'/content/drive/Shareddrives/SIADS - 694-695 Team Drive/models/spark_models/a1_{f_name}_model.sav'
      if not os.path.isdir(path_name):
        save_model(model, '/content/drive/Shareddrives/SIADS - 694-695 Team Drive/models/spark_models',
                    f_name, sklearn_model=False)
      elif os.path.isdir(path_name) and force == True:
        save_model(model, '/content/drive/Shareddrives/SIADS - 694-695 Team Drive/models/spark_models',
                    f_name, sklearn_model=False)
        print('Passed: ', model)
    except Exception as e:
      print(f'Failed because {e};', model)
  return

if __name__ == '__main__':
  import argparse
  import webbrowser
  import os, sys

  RANDOM_SEED = 655

  parser = argparse.ArgumentParser(description='Parse input and output directories.')
  parser.add_argument('--force', type=bool,
                      required=False, help='Provide the mlflow tracking password.')
  parser.add_argument('--mlflow_project', type=str,
                      required=True, help='Provide the mlflow tracking projectname.')
  parser.add_argument('--input_dataset', type=str,
                      required=True, help='Provide the mlflow tracking projectname.')
  parser.add_argument('--model_directory', type=str,
                      required=True, help='Provide the output model directory.')

  args = parser.parse_args()

  data = pd.read_csv(
    '/content/drive/Shareddrives/SIADS - 694-695 Team Drive/datasets/model_files/A1_B2_data.csv'
  )

  X = pyspark_column_prep(data)

  spark = SparkSession.builder \
  .master("local[*]") \
  .appName("Binary Buyer Prediction") \
  .getOrCreate()

  train_df = (spark.createDataFrame(X)
                   .withColumnRenamed('totals_transactions','label')
                   .withColumn('day', dayofmonth('date'))
                   .withColumn('year', year('date'))
                   .withColumn('month', month('date'))
                   .withColumn('week_day', dayofweek('date'))
                   .drop('date','socialEngagementType','buyers',
                         'Monetary','Frequency', 'repurchasers'))
  
  unbalanced_df = X.withColumn('label', when(col('label')==0.0, 0).otherwise(1.0))
  label_balance = (unbalanced_df.select('label','year')
                                .groupBy('label').count()
                                .withColumnRenamed('count','unbalanced_count'))

  downsample_df = downsampling(unbalanced_df)
  random_df = take_random_sample(unbalanced_df)

  X_train_unbalanced, X_test_unbalanced = unbalanced_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)
  X_train_dsample, X_test_dsample = downsample_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)
  X_train_random, X_test_random  = random_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)
  
  spark_datasets = {
    ('X_train_unbalanced', 'X_test_unbalanced'): (X_train_unbalanced, X_test_unbalanced),
    ('X_train_dsample', 'X_test_dsample'): (X_train_dsample, X_test_dsample),
    ('X_train_random', 'X_test_random'): (X_train_random, X_test_random)
  }

  spark_models ={
    'spark_lr': LogisticRegression(maxIter=100000, featuresCol='features',labelCol='label'),
    'spark_dt': classification.DecisionTreeClassifier(featuresCol="features", labelCol="label"),
    'spark_rf': RandomForestClassifier(featuresCol="features", labelCol="label")
  }

  param_grids ={
    'spark_lr': ParamGridBuilder()\
    .addGrid(spark_models['spark_lr'].regParam, [1.0, 2.0]) \
    .addGrid(spark_models['spark_lr'].elasticNetParam, [0.0, 0.5, 1]) \
    .build(),
    'spark_dt': ParamGridBuilder()\
    .addGrid(spark_models['spark_dt'].maxDepth, [1, 2, 3]) \
    .addGrid(spark_models['spark_dt'].minInfoGain, [0.0, .005, .01, .1]) \
    .build(),
    'spark_rf': ParamGridBuilder()\
    .addGrid(spark_models['spark_rf'].maxDepth, [1, 2, 3])\
    .addGrid(spark_models['spark_rf'].numTrees, [10, 20, 40]) \
    .build()
  }

  models, df, datasets = train_spark_models(spark_datasets, spark_models, param_grids)

  save_models(models, force=False)

  df = post_processing(df)
  df = df.reset_index()
  df.sort_values(['F1-Score','AUC'],ascending=False).to_csv('/content/drive/Shareddrives/SIADS - 694-695 Team Drive/results/a1_final_results.csv')
>>>>>>> Stashed changes
=======
>>>>>>> e71cd00eaf081f58f0d6bc90abf0653ee5c8f4ef
>>>>>>> 5b0eb06c41cea5fb7ef35dc5e473ab7dd93028c3
||||||| merged common ancestors
=======
<<<<<<< HEAD
'''Used for first ML model in our pipeline.'''
=======
X_train_spark = spark.createDataFrame(X_train)
X_val_spark = spark.createDataFrame(X_val)
y_train_spark = ps.from_pandas(y_train)
y_val_spark = ps.from_pandas(y_val)
>>>>>>> e71cd00eaf081f58f0d6bc90abf0653ee5c8f4ef
>>>>>>> 5b0eb06c41cea5fb7ef35dc5e473ab7dd93028c3
