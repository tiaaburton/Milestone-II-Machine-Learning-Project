"""
Creates a sample from the monthly GA dataset
"""
import os
import random
import pandas as pd
from utils import (append_sample, get_full_sample)
from argparse import ArgumentParser, Namespace

def create_sample_dataset(
    transactions_and_non_transactions_input_directory: str,
  ) -> pd.DataFrame:
  """
  This function does the following:
  1. Reads all transactions CSV from the input directory and join as one DF.
  2. Reads all non-transactions CSV from the input directory and samples 10%
  3. Join both results from Step 1 and 2 into one DF
  """
  transactions_directory = f"{transactions_and_non_transactions_input_directory}transactions/"
  transactions_monthly_files = os.listdir(transactions_directory)

  non_transactions_directory = f"{transactions_and_non_transactions_input_directory}non_transactions/"
  non_transactions_monthly_files = os.listdir(non_transactions_directory)

  # Get All Transactions
  transactions_df = pd.DataFrame({})
  for monthly_filename in transactions_monthly_files:
    transaction_file_name = f"{transactions_directory}{monthly_filename}"
    df = pd.read_csv(transaction_file_name)
    transactions_df = pd.concat([transactions_df, df], axis=0)

  outdir = f"../datasets/all_transactions_combined"
  if not os.path.exists(outdir):
    os.mkdir(outdir)

  transactions_df.to_csv(f"{outdir}/all_transactions.csv")

  # Sample Non-transactions
  non_transaction_files = [
    f"{non_transactions_directory}{monthly_filename}"
    for monthly_filename in non_transactions_monthly_files
  ]
  print(non_transaction_files)
  non_transactions_df = get_full_sample(non_transaction_files)

  # Combine both
  combined_df = pd.concat([non_transactions_df, transactions_df])
  return combined_df


def parse_args() -> Namespace:
    """parse arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "--input_directory",
        help="The Input Directory that contains csv files to be sampled from",
        type=str
    )
    parser.add_argument(
        "--output_file",
        help="The output file of the sampled dataset",
        type=str
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()

    input_directory = parsed_args.input_directory
    output_file = parsed_args.output_file

    print(input_directory, output_file)

    combined_df = create_sample_dataset(input_directory)
    combined_df.to_csv(output_file)

