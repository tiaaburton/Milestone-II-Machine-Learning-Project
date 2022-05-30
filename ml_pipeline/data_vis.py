import pandas as pd
from pandas_profiling import ProfileReport
from markupsafe import soft_unicode

def save_profiling_html(df, output_file):
  profile = ProfileReport(df)
  profile.to_file(output_file=output_file)
  return 'Opening dashboard file in new tab...'

if __name__ == '__main__':
  import argparse
  import webbrowser
  import os, sys

  parser = argparse.ArgumentParser(description='Parse input and output directories.')
  parser.add_argument('--input_directory', type=str,
                      required=True, help='Provide the input directory path.')
  parser.add_argument('--output_directory', type=str,
                      required=True, help='Provide the output directory path.')
  args = parser.parse_args()

  input_files = os.listdir(args.input_directory)
  for f in input_files:
    if f != 'changes_made.csv':
      df = pd.read_csv(args.input_directory + f)
      try:
        save_profiling_html(df, args.output_directory + f[:-4] + '_dashboard.html')
      except:
        pass
