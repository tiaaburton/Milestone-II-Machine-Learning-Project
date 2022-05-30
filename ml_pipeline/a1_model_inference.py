"""
Supervised Learning (A1): Conversion Likelihood Analysis - Model Inference.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import os
from argparse import ArgumentParser, Namespace

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

def get_clf_metrics(data:pyspark.sql.dataframe.DataFrame, true:str, pred:str):
  pred_df = data.select(true,pred).toPandas()
  recall = recall_score(pred_df['label'], pred_df['prediction'])
  f1Score = f1_score(pred_df['label'], pred_df['prediction'])
  precision = precision_score(pred_df['label'], pred_df['prediction'])
  return recall, f1Score, precision

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

def parse_args() -> Namespace:
  """parse arguments"""
  parser = ArgumentParser()
  parser.add_argument(
    "--input_models_directory",
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
  
  return parser.parse_args()


if __name__ == "__main__":
  parsed_args = parse_args()

  input_models_directory = parsed_args.input_models_directory
  output_directory = parsed_args.output_directory
  output_visualization_directory = parsed_args.output_visualization_directory

  print(input_models_directory, output_directory, output_visualization_directory)

  df = post_processing(df)
  df = df.reset_index()
  df.sort_values(['F1-Score','AUC'],ascending=False).to_csv('/content/drive/Shareddrives/SIADS - 694-695 Team Drive/results/a1_final_results.csv')

  # Your Code