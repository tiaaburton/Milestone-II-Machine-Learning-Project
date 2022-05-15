"""
A lot of these functions were ported over from `Extraction_Data_Sample.ipynb`
"""
import pandas as pd
import numpy as np
import os 
from datetime import datetime
from tqdm import tqdm
import random

import ast
import json
import re

FINAL_COLUMN_NAMES_PATH = \
  "/content/drive/Shareddrives/SIADS - 694-695 Team Drive/Notebooks/final_columns.txt"

def only_dict(d):
    """
    Convert json string representation of dictionary to a python dict
    """
    try:
      return ast.literal_eval(d)
    except:
      return extract_from_lists(d)
    else:
      return None

def extract_from_lists(row):
  for elem in row:
    if elem == None: return (None)
    else: return dict(elem)

def list_of_dicts(ld, field):
    """
    Create a mapping of the tuples formed after 
    converting json strings of list to a python list   
    """
    try:
      result = []
      for d in ast.literal_eval(ld):
        if d == None:
          continue
        else:
          result.append(d[field])
      return result
    except (TypeError, ValueError):
      return only_dict(ld)[field]
    else:
      return 'unable to retrieve'

def data_from_json(data, field):
  """
  Using the rows from the column and the field wanting to be selected
  the function finds the best method to parse the json code and return
  the nested value.
  """
  from json import JSONDecodeError

  try:
    json_data = json.dumps(data)
    json_field = json.loads(json_data.replace('\'', '\"')[1:-1])[field]
    return json_field
  except TypeError:
    return only_dict(data)[field]
  except JSONDecodeError:
    json_data = json.dumps(data)
    return list_of_dicts(json.loads(json_data), field)
  else:
    return None

def get_selected_columns(filename: str = FINAL_COLUMN_NAMES_PATH):
  """
  Filter out the necessary columns for our csv
  """
  columns = []
  with open(filename, 'r') as f:
    for ln in f.readlines():
      col = (ln.strip().split('\xa0\xa0')[1])
      groups = re.search(r"'(\w.+)'", col)
      columns.append(groups.group(1))
  return columns

def preprocessing(sample: pd.DataFrame, columns: list):
  """
  Given a sample DF, expand columns passed in args.
  """
  drop_cols = set()

  for col in tqdm(columns):
    try:
      splits = col.split('.')
      colName = ".".join(splits)
      if colName in sample.columns:
        continue
      elif len(splits) == 2:
        sample[colName] = sample[splits[0]].apply(lambda x: data_from_json(x, splits[1]))
      elif len(splits) == 3:
        prefix = ".".join(splits[:2])
        if prefix in sample.columns:
          sample[colName] = sample[prefix].apply(lambda x: data_from_json(x, splits[2]))
        else:
          sample[prefix] = sample[splits[0]].apply(lambda x: data_from_json(x, splits[1]))
          sample[colName] = sample[prefix].apply(lambda x: data_from_json(x, splits[2]))
        drop_cols.add(splits[0])
        drop_cols.add(prefix)
    except:
      splits = col.split('.')
      colName = ".".join(splits)
      if len(splits) == 3:
        prefix = ".".join(splits[:2])

        sample[colName] = \
          sample[prefix].apply(lambda x: only_dict(x)[splits[2]] if only_dict(x) is not None else None)
    else:
      continue
  
  return sample, drop_cols

def append_sample(
    filename: str,
    sample_size: int=10
  ) -> pd.DataFrame:
  """
  Samples a percentage of a CSV file
  """
  sample_size /= 100

  n = sum(1 for line in open(filename)) - 1 # number of records in file (excludes header)
  s = int(n * sample_size) # desired sample size
  skip = sorted(random.sample(range(1, n+1), n-s)) # the 0-indexed header will not be included in the skip list
  df = pd.read_csv(filename, skiprows=skip)

  return df

def get_full_sample(filenames: list) -> pd.DataFrame:
  """
  Given a list of filenames, return the samples of each
  of the filenames listed in one DF
  """
  full_sample = pd.DataFrame()

  for filename in tqdm(filenames):
    if len(full_sample) == 0:
      full_sample = append_sample(filename)
    else:
      file_sample = append_sample(filename)
      full_sample = pd.concat([full_sample, file_sample])

  return full_sample