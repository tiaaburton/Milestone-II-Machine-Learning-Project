import pandas as pd
import numpy as np
import pyarrow
import os 
import random
from datetime import datetime
from tqdm import tqdm

from google.cloud import bigquery
from google.colab import auth, files

import ast
import json
import re

def extract_on_dates(startdate, enddate, filename):
  import os.path
  
  bigquery_client = bigquery.Client()

  dates = pd.date_range(startdate, enddate,)
  formatted_dates = [date.strftime('%Y%m%d') for date in dates]
  dataset = pd.DataFrame()

  for idx, date in enumerate(formatted_dates):
    query = f'SELECT * FROM bigquery-public-data.google_analytics_sample.ga_sessions_{date}'
    # if idx % 5 == 0: print(date)
    df = bigquery_client.query(query).to_dataframe()
    dataset = pd.concat([dataset,df], sort=False)

  keep_cols = list(set([col.split('.')[0] for col in get_selected_columns() if col != 'clientId']))
  
  if os.path.isfile(filename):
    dataset[keep_cols].to_csv(filename, index=False, header=True, mode='o')
  else:
    dataset[keep_cols].to_csv(filename, index=False, header=True)

  return print(f'{startdate} to {enddate} data has been added to filename provided!')


def get_selected_columns(filename='/content/drive/Shareddrives/SIADS - 694-695 Team Drive/datasets/column_selection_text_files/initial_column_selection.txt'):
  """
  filename: """
  columns = []
  with open(filename, 'r') as f:
    for ln in f.readlines():
      col = (ln.strip().split('\xa0\xa0')[1])
      groups = re.search(r"'(\w.+)'", col)
      columns.append(groups.group(1))
  return columns


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Accept output directory and date arguments.')
  parser.add_argument('--output_directory', type=str,
                      required=True, help='Provide the output directory')
  parser.add_argument('--credentials_json', type=str,
                      required=True, help='Provide path to Google API key to use Big Query.')
  args = parser.parse_args()

  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.credentials_json
  directory = args.output_directory
  
  filename1 = directory + 'August 2016 Google Analytics Dataset.csv'
  filename2 = directory + 'September 2016 Google Analytics Dataset.csv'
  filename3 = directory + 'October 2016 Google Analytics Dataset.csv'
  filename4 = directory + 'November 2016 Google Analytics Dataset.csv'
  filename5 = directory + 'December 2016 Google Analytics Dataset.csv'
  filename6 = directory + 'January 2017 Google Analytics Dataset.csv'

  extraction_dates = [('08-01-2016','08-31-2016', filename1),
                      ('09-01-2016','09-30-2016', filename2),
                      ('10-01-2016','10-31-2016', filename3),
                      ('11-01-2016','11-30-2016', filename4),
                      ('12-01-2016','12-31-2016', filename5),
                      ('01-01-2017','01-31-2017', filename6)]

  for start, end, filename in extraction_dates:
    extract_on_dates(start, end, filename)


