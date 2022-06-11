"""
Supervised Learning (A2): Repurchaser Analysis - Model Training.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import os
import pickle
from argparse import ArgumentParser, Namespace

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

import warnings
# Grid Search has a lot of Warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV

import graphviz
from sklearn.tree import export_graphviz
import pydotplus

sns.set_style("darkgrid")
RANDOM_SEED = 655

classifiers = {
  'Dummy Classifier (Stratified)': DummyClassifier(
    strategy='stratified',
    random_state=RANDOM_SEED
  ),
  'Dummy Classifier (uniform)': DummyClassifier(
    strategy='uniform',
    random_state=RANDOM_SEED
  ),
  'Dummy Classifier (most frequent)': DummyClassifier(
    strategy='most_frequent',
    random_state=RANDOM_SEED
  ),
  'Logistic Regression': LogisticRegression(
    max_iter=10000,
    random_state=RANDOM_SEED
  ),
  'SVC': SVC(
    random_state=RANDOM_SEED
  ),
  'KNN': KNeighborsClassifier(),
  'Decision Trees': DecisionTreeClassifier(
    random_state=RANDOM_SEED   
  ),
  'Random Forests': RandomForestClassifier(
    random_state=RANDOM_SEED   
  )
}

grid_search_classifiers = {
  'Dummy Classifier (Stratified)': {
    "model": DummyClassifier(
      strategy='stratified',
      random_state=RANDOM_SEED
    ),
    "param_grid": {}
  },
  'Dummy Classifier (uniform)': {
    "model": DummyClassifier(
      strategy='uniform',
      random_state=RANDOM_SEED
    ),
    "param_grid": {}
  },
  'Dummy Classifier (most frequent)': {
    "model": DummyClassifier(
      strategy='most_frequent',
      random_state=RANDOM_SEED
    ),
    "param_grid": {}
  },
  'Logistic Regression': {
    "model": LogisticRegression(
      max_iter=10000,
      random_state=RANDOM_SEED
    ),
    "param_grid": {
      'penalty': ['l1','l2'],
      'C': [0.01, 0.1, 1, 10, 100, 1000],
      'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    }
  },
  'SVC': {
    "model": SVC(
      random_state=RANDOM_SEED
    ),
    "param_grid": {
      'C': [0.1, 1, 10, 100], 
      'gamma': [1, 0.1, 0.01, 0.001],
      'kernel': ['rbf', 'sigmoid'] # Removing Polynomial as it took 1+ hour per run
    }
  },
  'KNN': {
    "model": KNeighborsClassifier(),
    "param_grid": {
      'n_neighbors': [3, 5, 15, 25, 30],
      'weights': ['uniform', 'distance'],
      'metric': ['euclidean', 'manhattan'],
    },
  },
  'Decision Trees': {
    "model": DecisionTreeClassifier(
      random_state=RANDOM_SEED   
    ),
    "param_grid": {
      'criterion': ['gini', 'entropy'],
      'max_depth' : [None, 5, 10, 15, 30],
      'min_samples_split': [2, 4, 6, 8, 10],
      'min_samples_leaf': [1, 3, 5],
    }
  },
  'Random Forests': {
    "model": RandomForestClassifier(
      random_state=RANDOM_SEED   
    ),
    "param_grid": {
      'criterion': ['gini', 'entropy'],
      'max_depth': [None, 5, 10, 15, 30],
      'n_estimators': [200, 500],
      'max_features': ['sqrt', 'log2'],
    }
  },
}

optimal_balanced_classifiers = {
  'Logistic Regression': {
    "model": LogisticRegression(
      max_iter=10000,
      random_state=RANDOM_SEED,
      C=10,
      penalty='l2',
      solver='lbfgs',
    ),
    "model_name": "a2_lr_model.pkl",
  },
  'SVC': {
    "model": SVC(
      random_state=RANDOM_SEED,
      C=100, 
      gamma=0.001,
      kernel='rbf',
    ),
    "model_name": "a2_svc_model.pkl",
  },
  'KNN': {
    "model": KNeighborsClassifier(
      n_neighbors=3,
      metric='manhattan',
      weights='distance',
    ),
    "model_name": "a2_knn_model.pkl",
  },
  'Decision Trees': {
    "model": DecisionTreeClassifier(
      random_state=RANDOM_SEED,
      criterion='gini',
      max_depth=10,
      min_samples_leaf=5,
      min_samples_split=2,
    ),
    "model_name": "a2_dt_model.pkl",
  },
  'Random Forests': {
    "model": RandomForestClassifier(
      random_state=RANDOM_SEED,
      criterion='gini',
      max_depth=30,
      max_features='log2',
      n_estimators=200,
    ),
    "model_name": "a2_rf_model.pkl",
  },
}

features = [
  'socialEngagementType',
  'totals.hits',
  'totals.pageviews',
  'totals.timeOnSite',
  'device.browser_woe',
  'geoNetwork.country_woe',
  'trafficSource.source_woe',
  'totals.transactions',
  'hits.eCommerceAction.action_type',
  'hits.hour_ordinal',
  'totals.newVisits',
  'totals.bounces',
  'channelGrouping_Direct',
  'channelGrouping_Display',
  'channelGrouping_Organic Search',
  'channelGrouping_Paid Search',
  'channelGrouping_Referral',
  'channelGrouping_Social',
  'trafficSource.medium_affiliate',
  'trafficSource.medium_cpc',
  'trafficSource.medium_cpm',
  'trafficSource.medium_organic',
  'trafficSource.medium_referral',
  'trafficSource.isTrueDirect_code',
  'device.operatingSystem_Android',
  'device.operatingSystem_BlackBerry',
  'device.operatingSystem_Chrome OS',
  'device.operatingSystem_Firefox OS',
  'device.operatingSystem_Linux',
  'device.operatingSystem_Macintosh',
  'device.operatingSystem_Nintendo Wii',
  'device.operatingSystem_Samsung',
  'device.operatingSystem_Windows',
  'device.operatingSystem_Windows Phone',
  'device.operatingSystem_Xbox',
  'device.operatingSystem_iOS',
  'device.deviceCategory_mobile',
  'device.deviceCategory_tablet',
  'Recency',
  'Monetary',
  'Frequency',
  'buyers'
]

def generate_feature_correlation(df, img_path):
  """
  Generate Feature Correlation and save the image
  """
  corrmat = df.dropna().corr()
  top_corr_features = corrmat.index
  plt.figure(figsize=(20,20))

  # plot heat map
  g = sns.heatmap(
    df[top_corr_features].dropna().corr(),
    annot=True,
    cmap="RdYlGn"
  )
  fig = g.get_figure()
  fig.savefig(img_path)


def plot_decision_tree(clf, feature_names, class_names, output_visualization_directory):
  out_file_path = f"{output_visualization_directory}/adspy_temp.dot"
  export_graphviz(
    clf, 
    out_file=out_file_path, 
    feature_names=feature_names, 
    class_names=class_names, 
    filled=True, 
    impurity=False
  )
  with open(out_file_path) as f:
      dot_graph = f.read()

  graph_obj = graphviz.Source(dot_graph)
  png_bytes = graph_obj.pipe(format='png')
  with open(f"{output_visualization_directory}/dtree_pipe.png", "wb") as f:
      f.write(png_bytes)
      
  return graph_obj


def report_output(y_test, y_pred, print_info=True):
  """
  Reporting General Output
  """
  model_roc_auc_score = metrics.roc_auc_score(y_test, y_pred)
  model_f1_score = metrics.f1_score(y_test, y_pred)

  if print_info:
    print(metrics.classification_report(y_test, y_pred))
    print("roc_auc_score: ", model_roc_auc_score)
    print("f1 score: ", model_f1_score)

  return model_roc_auc_score, model_f1_score


def report_model_feature_importance(features, feature_importances):
  """
  Report feature importances from a given model
  """
  features_and_importances = list(zip(features, feature_importances))
  features_and_importances = sorted(features_and_importances, key=lambda x: -x[1])
  
  temp_res = {'feature': [], 'importance': []}
  for feature_name, feature_importance in features_and_importances:
    print(feature_name, feature_importance)
    temp_res['feature'].append(feature_name)
    temp_res['importance'].append(feature_importance)

  return temp_res


def plot_roc_auc_and_f1_scores(
    roc_auc_scores,
    f1_scores,
    classifier_labels,
    chart_title="Chart TItle",
    save_figure_path="../visualizations/a2_figure.png",
  ):
  """
  Plots the given ROC - AUC and F1 Scores
  """
  fig = plt.figure()
  
  # set width of bar
  barWidth = 0.25
  fig = plt.subplots(figsize=(24, 8))
  plt.grid(axis='x')
  # plt.grid(True)
  # plt.gca().xaxis.grid()
  # plt.gca().yaxis.grid()
  
  # set height of bar
  plt.ylim(0, 1)
  
  # Set position of bar on X axis
  br1 = np.arange(len(roc_auc_scores))
  br2 = [x + barWidth for x in br1]
  
  # Make the plot
  plt.bar(
    br1,
    roc_auc_scores, 
    color='green',
    width=barWidth,
    # edgecolor='grey',
    alpha=0.5,
    label='ROC AUC Score',
    zorder=3
  )
  plt.bar(
    br2,
    f1_scores,
    color='y',
    width=barWidth,
    # edgecolor='grey',
    alpha=0.5,
    label='F1 Score',
    zorder=3
  )
  
  # Adding Xticks
  plt.xlabel('Classifiers', fontweight='bold', fontsize=15)
  plt.ylabel('Scores', fontweight='bold', fontsize=15)
  plt.xticks(
    [r + barWidth/2 for r in range(len(roc_auc_scores))],
    classifier_labels
  )
  plt.title(chart_title, fontweight='bold', fontsize=17)
  
  plt.legend()

  plt.savefig(save_figure_path)
  plt.show()


def class_rebalance(df, features, target_class, output_visualization_directory):
  X = df[features]
  y = df[target_class]
  
  # rebalance
  smote = SMOTE(random_state=RANDOM_SEED)
  # fit predictor and target variable
  X_smote, y_smote = smote.fit_resample(X, y)
  
  fig = plt.figure()
  fig.set_size_inches(8, 3)

  ax1 = plt.subplot(121)
  ax1 = sns.countplot(df[target_class])
  ax1.set_xlabel("Repurchasers", fontsize=12)
  ax1.set_xticklabels(['0', '1'])
  ax1.set_ylabel("Count", fontsize=12)
  ax1.set_title("Before rebalancing", fontsize=14)
  ax1.set_facecolor('#EAEAF2')
  
  ax2 = plt.subplot(122)
  ax2 = sns.countplot(y_smote)
  ax2.set_xlabel("Repurchasers", fontsize=12)
  ax2.set_xticklabels(['0', '1'])
  ax2.set_ylabel(" ", fontsize=12)
  ax2.set_title("After rebalancing", fontsize=14)
  ax2.set_facecolor('#EAEAF2')
  
  fig.savefig(
    f"{output_visualization_directory}/rebalanced_repurchasers.png", 
    bbox_inches='tight'
  )
  
  rebalanced_df = pd.concat([X_smote, y_smote], axis=1)

  return rebalanced_df


def grid_search_clf_runs(grid_search_classifiers, X, y):
  """
  Runs the grid search classifiers on a given X and y
  """
  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    random_state=RANDOM_SEED
  )

  finetuned_roc_auc_scores = []
  finetuned_f1_scores = []
  clf_best_params = {}
  
  for clf_name, clf_info in grid_search_classifiers.items():
    print("Beginning", clf_name)
    clf_model = clf_info["model"]
    param_grid = clf_info["param_grid"]
  
    clf = GridSearchCV(
      clf_model,
      param_grid,
      scoring='f1_macro',
    )
    clf.fit(X_train, y_train)
  
    clf_best_params[clf_name] = clf.best_params_
  
    # Using the best parameters, let's get the ROC/AUC and F1 Scores
    y_pred = clf.predict(X_test)
    model_roc_auc_score, model_f1_score = report_output(y_test, y_pred, print_info=False)
    finetuned_roc_auc_scores.append(model_roc_auc_score)
    finetuned_f1_scores.append(model_f1_score)

  return finetuned_roc_auc_scores, finetuned_f1_scores, clf_best_params


def run_grid_search(df, chart_title, output_visualization_filename):
  """
  WARNING: THIS CAN TAKE AWHILE
  Runs Grid Search
  """
  df_input = df.dropna()
  X = df_input[features]  # independent columns
  y = df_input.iloc[:,-1]    # target column: repurchasers

  finetuned_roc_auc_scores, finetuned_f1_scores, clf_best_params = \
    grid_search_clf_runs(grid_search_classifiers, X, y)

  plot_roc_auc_and_f1_scores(
    finetuned_roc_auc_scores,
    finetuned_f1_scores,
    list(grid_search_classifiers.keys()),
    chart_title=chart_title,
    save_figure_path=output_visualization_filename
  )


def train_optimal_models(df_rebalanced, output_directory):
  """
  Trains the models with optimal parameters and saves it to `output_directory`
  """
  df_rebalanced = df_rebalanced.dropna()
  X = df_rebalanced[features]   # independent columns
  y = df_rebalanced.iloc[:,-1]  # target column: repurchasers
  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    random_state=RANDOM_SEED
  )
  optimal_roc_auc_scores = []
  optimal_f1_scores = []

  for clf_name, clf_info in optimal_balanced_classifiers.items():
    print("Beginning", clf_name)
    clf = clf_info["model"].fit(X_train, y_train)

    model_filename = clf_info['model_name']
    model_savepath = f"{output_directory}/a2_models/"
    if not os.path.exists(model_savepath):
      os.mkdir(model_savepath)

    with open(f"{model_savepath}/{model_filename}", 'wb') as pkl_file:
      pickle.dump(clf, pkl_file)

    y_pred = clf.predict(X_test)

    # Using the best parameters, let's get the ROC/AUC and F1 Scores
    model_roc_auc_score, model_f1_score = report_output(y_test, y_pred, print_info=False)
    optimal_roc_auc_scores.append(model_roc_auc_score)
    optimal_f1_scores.append(model_f1_score)
  
  plot_roc_auc_and_f1_scores(
    optimal_roc_auc_scores,
    optimal_f1_scores,
    list(optimal_balanced_classifiers.keys()),
    chart_title="Optimal Models Trained on Finetuned Models Accuracy Scores"
  )


def report_important_features(df_rebalanced, output_result_directory):
  """
  Gets the important features and writes it into a CSV
  """
  df_rebalanced = df_rebalanced.dropna()
  X = df_rebalanced[features]   # independent columns
  y = df_rebalanced.iloc[:,-1]  # target column: repurchasers

  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    random_state=RANDOM_SEED
  )

  optimal_dt_clf = optimal_balanced_classifiers["Decision Trees"]["model"].fit(
    X_train, 
    y_train
  )

  dt_feature_importances = list(optimal_dt_clf.feature_importances_)
  dt_feature_importance_dict = report_model_feature_importance(
    features, 
    dt_feature_importances
  )

  feature_importance_df = pd.DataFrame.from_dict(dt_feature_importance_dict)
  feature_importance_df.to_csv(
    f"{output_result_directory}/a2_feature_importance.csv"
  )

  return dt_feature_importances


def prune_model_feature_importance(
    features,
    feature_importances,
    threshold=0.005
  ):
  """
  Return the important features from a given model and a threshold
  """
  features_and_importances = list(zip(features, feature_importances))
  features_and_importances = sorted(features_and_importances, key=lambda x: -x[1])
  
  temp_res = {'feature': [], 'importance': []}
  new_features = []
  for feature_name, feature_importance in features_and_importances:
    if feature_importance >= threshold:
      print(feature_name, feature_importance)
      new_features.append(feature_name)
      temp_res['feature'].append(feature_name)
      temp_res['importance'].append(feature_importance)

  return new_features, temp_res


def rerun_model_with_pruned_features(
  df_rebalanced, 
  features, 
  dt_feature_importances,
  figure_output_path
):
  """
  Rerun the model with only the top features
  """
  feature_subset, feature_subset_dict = prune_model_feature_importance(
    features,
    dt_feature_importances,
    threshold=0.01
  )

  df_rebalanced = df_rebalanced.dropna()
  X = df_rebalanced[feature_subset]   # independent columns
  y = df_rebalanced.iloc[:,-1]  # target column: repurchasers

  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    random_state=RANDOM_SEED
  )

  feat_subset_optimal_dt_clf = optimal_balanced_classifiers["Decision Trees"]["model"].fit(X_train, y_train)
  feat_subset_optimal_roc_auc_scores = []
  feat_subset_optimal_f1_scores = []
  feat_subset_optimal_clf_best_params = {}

  for clf_name, clf_info in optimal_balanced_classifiers.items():
    print("Beginning", clf_name)
    clf = clf_info["model"].fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Using the best parameters, let's get the ROC/AUC and F1 Scores
    model_roc_auc_score, model_f1_score = report_output(y_test, y_pred, print_info=False)
    feat_subset_optimal_roc_auc_scores.append(model_roc_auc_score)
    feat_subset_optimal_f1_scores.append(model_f1_score)

  plot_roc_auc_and_f1_scores(
    feat_subset_optimal_roc_auc_scores,
    feat_subset_optimal_f1_scores,
    list(optimal_balanced_classifiers.keys()),
    chart_title="Finetuned, Rebalanced Dataset, and Feature Pruned Models",
    save_figure_path=figure_output_path
  )


def visualize_decision_tree(
    df_rebalanced,
    features, 
    dt_feature_importances,
    output_visualization_directory,
    max_depth=5
  ):
  """
  Visualize a decision tree
  """
  feature_subset, feature_subset_dict = prune_model_feature_importance(
    features,
    dt_feature_importances,
    threshold=0.01
  )

  df_rebalanced = df_rebalanced.dropna()
  X = df_rebalanced[feature_subset]   # independent columns
  y = df_rebalanced.iloc[:,-1]  # target column: repurchasers

  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    random_state=RANDOM_SEED
  )

  dt_fun_clf = DecisionTreeClassifier(
    random_state=RANDOM_SEED,
    criterion='gini',
    max_depth=max_depth,
    min_samples_leaf=5,
    min_samples_split=2,
  ).fit(X_train, y_train)

  y_pred = dt_fun_clf.predict(X_test)

  # Using the best parameters, let's get the ROC/AUC and F1 Scores
  model_roc_auc_score, model_f1_score = report_output(
    y_test, 
    y_pred, 
    print_info=False
  )
  print("ROC AUC Score", model_roc_auc_score)
  print("F1 Score", model_f1_score)

  plot_decision_tree(
    dt_fun_clf, 
    list(X.columns), 
    ['Not Repurchaser', 'Repurchaser'],
    output_visualization_directory
  )


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
    "--output_result_directory",
    help="The output result directory of all CSVs created",
    type=str
  )
  parser.add_argument(
    "--output_visualization_directory",
    help="The output directory of all visualizations created",
    type=str
  )
  parser.add_argument(
    "--full_run",
    help="Full run includes: All Grid Search",
    type=bool,
    default=False
  )
  
  return parser.parse_args()


if __name__ == "__main__":
  parsed_args = parse_args()

  input_dataset = parsed_args.input_dataset
  output_directory = parsed_args.output_directory
  output_result_directory = parsed_args.output_result_directory
  output_visualization_directory = parsed_args.output_visualization_directory
  full_run = parsed_args.full_run

  print(input_dataset, output_directory, output_result_directory, output_visualization_directory)

  df = pd.read_csv("../datasets/model_files/A2_return_data.csv")

  generate_feature_correlation(
    df,
    f"{output_visualization_directory}/a2_feature_correlation_heatmap.png"
  )

  if full_run:
    run_grid_search(
      df,
      "Finetuned Models",
      f"{output_visualization_directory}/a2_finetuned_models.png"
    )
  
  df_rebalanced = class_rebalance(
    df.dropna(), 
    features, 
    "repurchasers",
    output_visualization_directory
  )

  if full_run:
    run_grid_search(
      df,
      "Finetuned and Rebalanced Dataset Models",
      f"{output_visualization_directory}/a2_finetuned_rebalanced_models.png"
    )

  train_optimal_models(df_rebalanced, output_directory)

  dt_feature_importances = \
    report_important_features(df_rebalanced, output_result_directory)

  rerun_model_with_pruned_features(
    df_rebalanced, 
    features, 
    dt_feature_importances,
    f"{output_visualization_directory}/a2_finetuned_feature_pruned_models.png"
  )

  visualize_decision_tree(
    df_rebalanced,
    features, 
    dt_feature_importances,
    output_visualization_directory,
    max_depth=5
  )
    
