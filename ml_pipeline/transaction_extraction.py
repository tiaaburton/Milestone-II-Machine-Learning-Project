"""
Creates a sample from the monthly GA dataset
"""
import os
import pandas as pd
from utils import (get_selected_columns, preprocessing)
from argparse import ArgumentParser, Namespace

def parse_transactions_from_df(
    df_input: pd.DataFrame, 
    intermediate_output_directory_path: str,
    monthly_file: str
  ):
  """
  Given an input of a DF from GA365, extract all of the rows
  that have `totals.totalTransactionRevenue` present within the `totals` column.
  """
  columns_of_interest = get_selected_columns()
  curr_df, _ = preprocessing(
    df_input,
    columns_of_interest
  )

  # Transactions
  # Create outdir if doesn't exist
  transactions_outdir = f"{intermediate_output_directory_path}transactions"
  if not os.path.exists(transactions_outdir):
    os.mkdir(transactions_outdir)

  transactions_file_output_path = \
    f'{transactions_outdir}/transactions_from_{monthly_file}'
  transactions_df = curr_df[curr_df['totals.totalTransactionRevenue'].notnull()]
  print("Also writing transactions into:", transactions_file_output_path)
  transactions_df.to_csv(transactions_file_output_path)

  # Non-Transactions
  # Create outdir if doesn't exist
  non_transactions_outdir = f"{intermediate_output_directory_path}non_transactions"
  if not os.path.exists(non_transactions_outdir):
    os.mkdir(non_transactions_outdir)

  non_transactions_file_output_path = \
    f'{non_transactions_outdir}/non_transactions_from_{monthly_file}'
  non_transactions_df = curr_df[~curr_df['totals.totalTransactionRevenue'].notnull()]
  print(
    "Also writing non-transactions into:", 
    non_transactions_file_output_path
  )
  non_transactions_df.to_csv(non_transactions_file_output_path)

  return transactions_df


def read_input_directory(
    input_directory_path: str,
    intermediate_output_directory_path: str
  ):
  """
  Traverse input directory files and parse transactions
  """
  monthly_files = os.listdir(input_directory_path)
  res_df = pd.DataFrame({})

  # Completed:
  # August 2016 Google Analytics Dataset.csv
  # September 2016 Google Analytics Dataset.csv
  # October 2016 Google Analytics Dataset.csv
  # November 2016 Google Analytics Dataset.csv
  # December 2016 Google Analytics Dataset.csv
  # January 2017 Google Analytics Dataset.csv
  for monthly_file in monthly_files:
    curr_monthly_file_path = f'{input_directory_path}{monthly_file}'
    print("Beginning:", curr_monthly_file_path)
    df = pd.read_csv(curr_monthly_file_path)

    parsed_df = parse_transactions_from_df(
      df,
      intermediate_output_directory_path,
      monthly_file,
    )

    res_df = pd.concat([res_df, parsed_df], axis=0)
    print("Completed:", curr_monthly_file_path)

  return res_df


def parse_args() -> Namespace:
  """parse arguments"""
  parser = ArgumentParser()
  parser.add_argument(
    "--input_directory",
    help="The Input Directory that contains csv files to be sampled from",
    type=str
  )
  parser.add_argument(
    "--output_directory",
    help="The output file of all transactions extracted",
    type=str
  )
  
  return parser.parse_args()


if __name__ == "__main__":
  parsed_args = parse_args()

  input_directory = parsed_args.input_directory
  output_directory = parsed_args.output_directory

  print(input_directory, output_directory)

  res_df = read_input_directory(
    input_directory,
    output_directory
  )


"""
Output Logs:
../datasets/monthly_partitioned_data/ ../datasets/monthly_partitioned_data_transactions/ ../datasets/all_transactions.csv
Beginning: ../datasets/monthly_partitioned_data/August 2016 Google Analytics Dataset.csv
tcmalloc: large alloc 1073741824 bytes == 0x45072000 @  0x7f0644a602a4 0x7f0632eae9a5 0x7f0632eafcc1 0x7f0632eb169e 0x7f0632e8250c 0x7f0632e8f399 0x7f0632e7797a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
tcmalloc: large alloc 2147483648 bytes == 0x85072000 @  0x7f0644a602a4 0x7f0632eae9a5 0x7f0632eafcc1 0x7f0632eb169e 0x7f0632e8250c 0x7f0632e8f399 0x7f0632e7797a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
tcmalloc: large alloc 4294967296 bytes == 0x105072000 @  0x7f0644a602a4 0x7f0632eae9a5 0x7f0632eafcc1 0x7f0632eb169e 0x7f0632e8250c 0x7f0632e8f399 0x7f0632e7797a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
100% 37/37 [2:12:58<00:00, 215.64s/it]
Also writing transactions into: ../datasets/monthly_partitioned_data_transactions/transactions/transactions_from_August 2016 Google Analytics Dataset.csv
Also writing non-transactions into: ../datasets/monthly_partitioned_data_transactions/non_transactions/non_transactions_from_August 2016 Google Analytics Dataset.csv
Completed: ../datasets/monthly_partitioned_data/August 2016 Google Analytics Dataset.csv

Beginning: ../datasets/monthly_partitioned_data/September 2016 Google Analytics Dataset.csv
tcmalloc: large alloc 1073741824 bytes == 0x450e8000 @  0x7f74c6fb92a4 0x7f74b542b9a5 0x7f74b542ccc1 0x7f74b542e69e 0x7f74b53ff50c 0x7f74b540c399 0x7f74b53f497a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
tcmalloc: large alloc 2147483648 bytes == 0x850e8000 @  0x7f74c6fb92a4 0x7f74b542b9a5 0x7f74b542ccc1 0x7f74b542e69e 0x7f74b53ff50c 0x7f74b540c399 0x7f74b53f497a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
tcmalloc: large alloc 4294967296 bytes == 0x1050e8000 @  0x7f74c6fb92a4 0x7f74b542b9a5 0x7f74b542ccc1 0x7f74b542e69e 0x7f74b53ff50c 0x7f74b540c399 0x7f74b53f497a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
100% 37/37 [1:57:42<00:00, 190.87s/it]
Also writing transactions into: ../datasets/monthly_partitioned_data_transactions/transactions/transactions_from_September 2016 Google Analytics Dataset.csv
Also writing non-transactions into: ../datasets/monthly_partitioned_data_transactions/non_transactions/non_transactions_from_September 2016 Google Analytics Dataset.csv
Completed: ../datasets/monthly_partitioned_data/September 2016 Google Analytics Dataset.csv

Beginning: ../datasets/monthly_partitioned_data/October 2016 Google Analytics Dataset.csv
tcmalloc: large alloc 2147483648 bytes == 0x13ace0000 @  0x7f74c6fb92a4 0x7f74b542b9a5 0x7f74b542ccc1 0x7f74b542e69e 0x7f74b53ff50c 0x7f74b540c399 0x7f74b53f497a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
100% 37/37 [1:16:09<00:00, 123.49s/it]
Also writing transactions into: ../datasets/monthly_partitioned_data_transactions/transactions/transactions_from_October 2016 Google Analytics Dataset.csv
Also writing non-transactions into: ../datasets/monthly_partitioned_data_transactions/non_transactions/non_transactions_from_October 2016 Google Analytics Dataset.csv
Completed: ../datasets/monthly_partitioned_data/October 2016 Google Analytics Dataset.csv

Beginning: ../datasets/monthly_partitioned_data/November 2016 Google Analytics Dataset.csv
tcmalloc: large alloc 2147483648 bytes == 0x2058e8000 @  0x7f74c6fb92a4 0x7f74b542b9a5 0x7f74b542ccc1 0x7f74b542e69e 0x7f74b53ff50c 0x7f74b540c399 0x7f74b53f497a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
100% 37/37 [1:17:09<00:00, 125.12s/it]
Also writing transactions into: ../datasets/monthly_partitioned_data_transactions/transactions/transactions_from_November 2016 Google Analytics Dataset.csv
Also writing non-transactions into: ../datasets/monthly_partitioned_data_transactions/non_transactions/non_transactions_from_November 2016 Google Analytics Dataset.csv
Completed: ../datasets/monthly_partitioned_data/November 2016 Google Analytics Dataset.csv

Beginning: ../datasets/monthly_partitioned_data/December 2016 Google Analytics Dataset.csv
tcmalloc: large alloc 2147483648 bytes == 0xdd482000 @  0x7f74c6fb92a4 0x7f74b542b9a5 0x7f74b542ccc1 0x7f74b542e69e 0x7f74b53ff50c 0x7f74b540c399 0x7f74b53f497a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
100% 37/37 [1:23:59<00:00, 136.20s/it]
Also writing transactions into: ../datasets/monthly_partitioned_data_transactions/transactions/transactions_from_December 2016 Google Analytics Dataset.csv
Also writing non-transactions into: ../datasets/monthly_partitioned_data_transactions/non_transactions/non_transactions_from_December 2016 Google Analytics Dataset.csv
Completed: ../datasets/monthly_partitioned_data/December 2016 Google Analytics Dataset.csv

Beginning: ../datasets/monthly_partitioned_data/January 2017 Google Analytics Dataset.csv
tcmalloc: large alloc 1073741824 bytes == 0x449b6000 @  0x7f0ab31b02a4 0x7f0aa16199a5 0x7f0aa161acc1 0x7f0aa161c69e 0x7f0aa15ed50c 0x7f0aa15fa399 0x7f0aa15e297a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
tcmalloc: large alloc 2147483648 bytes == 0x849b6000 @  0x7f0ab31b02a4 0x7f0aa16199a5 0x7f0aa161acc1 0x7f0aa161c69e 0x7f0aa15ed50c 0x7f0aa15fa399 0x7f0aa15e297a 0x59afff 0x515655 0x549e0e 0x593fce 0x511e2c 0x549576 0x593fce 0x511e2c 0x593dd7 0x5118f8 0x549576 0x4bcb19 0x5134a6 0x549e0e 0x593fce 0x548ae9 0x51566f 0x593dd7 0x5118f8 0x549576 0x604173 0x5f5506 0x5f8c6c 0x5f9206
100% 37/37 [57:37<00:00, 93.44s/it] 
Also writing transactions into: ../datasets/monthly_partitioned_data_transactions/transactions/transactions_from_January 2017 Google Analytics Dataset.csv
Also writing non-transactions into: ../datasets/monthly_partitioned_data_transactions/non_transactions/non_transactions_from_January 2017 Google Analytics Dataset.csv
Completed: ../datasets/monthly_partitioned_data/January 2017 Google Analytics Dataset.csv
"""