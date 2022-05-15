import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from markupsafe import soft_unicode
import ast

def save_profiling_html(df, output_file):
  profile = ProfileReport(df)
  profile.to_file(output_file=output_file)
  return 'Opening dashboard file in new tab...'

def drop_columns(df):
  with open('/content/drive/Shareddrives/SIADS - 694-695 Team Drive/datasets/column_selection_text_files/final_column_selection.txt','r') as columns:
    keep=[]
    for ln in columns.readlines():
      keep.append(ln.replace('\n',''))
  return df[keep]

def set_datatypes(df):
  chg_cols = ['hits.type','hits.hour','hits.minute']
  for col in chg_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x))
  return df

if __name__ == '__main__':
  import argparse
  import webbrowser
  import os

  parser = argparse.ArgumentParser(description='Parse input and output files.')
  parser.add_argument('-input', '--input_file', type=str,
                      required=True, help='Provide the input file path.')
  parser.add_argument('-output', '--output_file', type=str,
                      required=True, help='Provide the output file path.')
  parser.add_argument('-dashboard', '--dashboard_file', type=str,
                      required=True, help='Provide the dashboard file path.')
  args = parser.parse_args()
  
  sample = pd.read_csv(args.input_file)
  sample = drop_columns(sample)
  sample = set_datatypes(sample)
  sample.to_csv(args.output_file, header=True)

  try:
    save_profiling_html(sample, args.dashboard_file)
  except:
    pass
