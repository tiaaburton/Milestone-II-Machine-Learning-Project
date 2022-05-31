"""
Supervised Learning (A1): Conversion Likelihood Analysis - Model Training.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import os
from argparse import ArgumentParser, Namespace

# REQUIREMENTS FILE
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

import pyspark
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml import *
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_classif

import chart_studio.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

sns.set_style("darkgrid")
RANDOM_SEED = 655

def pyspark_column_prep(dataset):
  """Returns column names with underscores for PySpark ready format.
     dataset: pd.DataFrame"""
  for col in dataset.columns:
    if '.' in col:
      repl = col.replace('.','_').replace(' ','_')
      dataset.rename(columns={col: repl},inplace=True)
  return dataset

def return_samples(df):
  unbalanced_df = df.withColumn('label', when(col('label')==0.0, 0).otherwise(1.0))
  downsample_df = downsampling(unbalanced_df)
  random_df = take_random_sample(unbalanced_df)
  return unbalanced_df, downsample_df, random_df

def downsampling(data):
  label_balance = data.groupBy('label').count().withColumnRenamed('count','unbalanced_count')
  df = data.withColumn('rand_col', rand())
  balanced_data = df.withColumn(
      "row_num",row_number().over(Window.partitionBy("label")\
                                  .orderBy("rand_col"))).filter(col("row_num")<=label_balance.head(2)[1][1])\
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

def save_model(df, idx, model, file_loc, file_name):
  dataset = df.iloc[idx]['Dataset'].split("_")[-1]
  col_num = df.iloc[idx]['Col_Num']
  f_name = df.iloc[idx]['Model'] + f'_run{idx}_{dataset}_{col_num}_col'
  filename = file_loc + file_name
  try:
    model.save(filename)
  except:
    model.write().overwrite().save(filename)
  return

def train_spark_models(spark_datasets, spark_models, param_grids, filters, models):

  trained_models = []
  df = pd.DataFrame()
  datasets = []

  for model in models:
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

          df2 = pd.DataFrame([[model, train_name, filter, len(filter), train_eval, train_recall, 
                               train_precision, train_f1Score, best_params, start_time, end_time]],
                      columns=['Model', 'Dataset', 'Column_Filter', 'Col_Num',
                               'AUC', 'Recall', 'Precision', 'F1-Score', 'Best_Params',
                               'Start_Time', 'End_Time'])
          df3 = pd.DataFrame([[model, test_name, filter, len(filter), test_eval, test_recall, 
                               test_precision, test_f1Score, best_params, start_time, end_time]],
                      columns=['Model', 'Dataset','Column_Filter', 'Col_Num',
                               'AUC', 'Recall', 'Precision', 'F1-Score', 'Best_Params',
                               'Start_Time', 'End_Time'])
          if len(df) == 0:
            df = pd.concat([df2, df3], ignore_index=True)
          else:
            df = pd.concat([df, df2, df3], ignore_index=True)

  return trained_models, df, datasets

def get_clf_metrics(data:pyspark.sql.dataframe.DataFrame, true:str, pred:str):
  pred_df = data.select(true,pred).toPandas()
  recall = recall_score(pred_df['label'], pred_df['prediction'])
  f1Score = f1_score(pred_df['label'], pred_df['prediction'])
  precision = precision_score(pred_df['label'], pred_df['prediction'])
  return recall, f1Score, precision

def post_processing(df):
  for idx,param_grid in enumerate(df.Best_Params):
    for param_str in (param_grid[3:-2].split(',\n\t')):
      param = param_str.split(':')
      param[0] = param[0].split('-')[-1]

      if param[0] not in df.columns:
        df[param[0]] = np.nan
      elif param[0] in df.columns and len(param) > 1: 
        df[param[0]].iloc[idx] = param[1]
  return df

def save_models(df, models:list, directory:str, force:bool=False):
  if force == True:
    for idx,model in enumerate(models):
      try:
        col_count = len(df.iloc[idx]['Column_Filter'])
        f_name = df.iloc[idx]['Model'] + f'_{col_count}MIcol_run{idx}'
        path_name = directory + f'a1_{f_name}_model.sav'
        if force == True:
          save_model(df, idx, model, directory, f_name)
          print('Passed: ', model)
      except Exception as e:
        print(f'Failed because {e};', model)
      else:
        print('Unable to save models. Troubleshooting needed.')
  return

def parse_args() -> Namespace:
  """parse arguments"""
  parser = ArgumentParser()
  parser.add_argument(
    "--input_dataset",
    help="The CSV Input dataset",
    type=str
  )
  parser.add_argument(
    "--output_directory",
    help="The output directory of all models created",
    type=str
  )
  parser.add_argument(
    "--output_visualization_directory",
    help="The output directory of all visualizations created",
    type=str
  )
  parser.add_argument(
    "--output_result_directory",
    help="The output directory of all results created",
    type=str
  )
  parser.add_argument(
    "--save", "-s",
    help="Force the models to be stored.",
    type=bool
  )
  return parser.parse_args()

if __name__ == "__main__":
  parsed_args = parse_args()

  input_dataset = parsed_args.input_dataset
  output_directory = parsed_args.output_directory
  output_visualization_directory = parsed_args.output_visualization_directory
  output_result_directory = parsed_args.output_result_directory
  save = parsed_args.save

  print(input_dataset, output_directory, output_visualization_directory)
  
  spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Binary Buyer Prediction") \
    .getOrCreate()

  spark.newSession()

  a1_b2_dataset = pd.read_csv(input_dataset)
  spark_ready_X = pyspark_column_prep(a1_b2_dataset)

  train = spark_ready_X[
    spark_ready_X.columns[~spark_ready_X.columns.isin([
      'Monetary',
      'buyers',
      'Frequency',
      'repurchasers',
      'Recency',
      'totals_transactions',
      'date',
      'fullVisitorId'])
      ]
  ].fillna(0.0)

  target = a1_b2_dataset[['totals_transactions']]
  target = target.totals_transactions.fillna(0.0).apply(lambda trans: trans if trans == 0.0 else 1.0)
  target = pd.Categorical(target)
  target = pd.Series(target)

  train_spark = (spark.createDataFrame(spark_ready_X)
                    .withColumn('label', col('totals_transactions')))
  
  unbalanced_df, downsample_df, random_df = return_samples(train_spark)

  X_train_unbalanced, X_test_unbalanced = unbalanced_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)
  X_train_dsample, X_test_dsample = downsample_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)
  X_train_random, X_test_random  = random_df.randomSplit([0.9, 0.1], seed=RANDOM_SEED)

  filter1,mutual_info = get_mutual_info_features(train, target, 0.0)
  filter2,_ = get_mutual_info_features(train, target, .001)
  filter3,_ = get_mutual_info_features(train, target, .05)

  mutual_info = pd.DataFrame(mutual_info, columns=['mutual_info_score', 'column_name'])
  mutual_info.to_csv(output_result_directory + 'a1_mutual_info_results.csv')

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

  models, df, datasets = train_spark_models(spark_datasets, spark_models, param_grids,
                    [filter1, filter2, filter3], ['spark_lr', 'spark_dt', 'spark_rf'])
  df.sort_values(['F1-Score','AUC'],ascending=False)\
    .to_csv(output_result_directory +'a1_final_results.csv')

  save_models(df, models, output_directory, force=save)

  titles = ["Likelihood to Buy Binary Classification",
            "Logistic Regression Params vs F1-Score in Likelihood to Buy Binary Classification",
            "Logistic Regression F1-Score in Likelihood to Buy Binary Classification",
            "Logistic Regression Recall Scores in Likelihood to Buy Binary Classification",
            "Logistic Regression Precision Scores in Likelihood to Buy Binary Classification"]

  fig1 = px.scatter(df, x="AUC", y="F1-Score", facet_col="Col_Num", facet_row="Model", 
                 title=titles[0], color='Dataset'
                 )
  fig2 = px.scatter_3d(df[df['Model']=='spark_lr'], x="regParam", y="elasticNetParam",
                       z="F1-Score", title=titles[1], color='Dataset', symbol='Col_Num'
          )
  fig3 = px.bar(df[df['Model']=='spark_lr'], x="index", y="F1-Score", 
                color="Dataset", text_auto='.2', title=titles[2], 
          )
  fig4 = px.bar(df[df['Model']=='spark_lr'], x="index", y="Recall", 
                color="Dataset", text_auto='.2', title=titles[3], 
          )
  fig5 = px.bar(df[df['Model']=='spark_lr'], x="index", y="Precision", 
                color="Dataset", text_auto='.2', title=titles[4], 
          )
  
  for idx,fig in enumerate([fig1, fig2, fig3, fig4, fig5]):
    file_name = output_visualization_directory + titles[idx]
    fig.write_html(file_name + '.html')
    fig.write_image(file_name + '.png')

  spark.sparkContext.stop()