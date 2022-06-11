data_extraction.py:
  python data_extraction.py \
    --output_directory ../datasets/monthly-partitioned-data/ \
    --credentials_json ../credentials/compact-scene-317315-56e479e9e148.json

transaction_extraction.py: ../datasets/monthly-partitioned-data/
  python transaction_extraction.py \
    --input_directory ../datasets/monthly_partitioned_data/ \
    --output_directory ../datasets/monthly_partitioned_data_transactions/

create_sample_dataset.py: ../datasets/monthly_partitioned_data_transactions/
  python create_sample_dataset.py \
    --input_directory ../datasets/monthly_partitioned_data_transactions/ \
    --output_file ../datasets/sample_dataset.csv

clean_dataset.py: ../datasets/sample_dataset.csv
  python clean_dataset.py \
    --input_file ../datasets/sample_dataset.csv \
    --output_file ../datasets/cleaned_dataset.csv \
    --dashboard_file ../datasets/dashboard_files/cleaned_dataset_dashboard.html

data_prep.py: ../datasets/cleaned_dataset.csv
  python data_prep.py \
    --input_file ../datasets/cleaned_dataset.csv \
    --output_directory ../datasets/model_files

data_vis.py: ../datasets/model_files
  python data_vis.py \
    --input_directory ../datasets/model_files/ \
    --output_directory ../datasets/dashboard_files/

a1_model_training.py: ../datasets/model_files/A1_B2_data.csv
  python a1_model_training.py \
    --input_dataset ../datasets/model_files/A1_B2_data.csv \
    --output_directory ../models/spark_models/ \
    --output_result_directory ../results/ \
    --output_visualization_directory ../visualizations/

a2_model_training.py: ../datasets/model_files/A2_return_data.csv
  python a2_model_training.py \
    --input_dataset ../datasets/model_files/A2_return_data.csv \
    --output_directory ../models/ \
    --output_result_directory ../results/ \
    --output_visualization_directory ../visualizations/

B2_B1_model_training.py:
  python B2_B1_model_training.py