X_train_spark = spark.createDataFrame(X_train)
X_val_spark = spark.createDataFrame(X_val)
y_train_spark = ps.from_pandas(y_train)
y_val_spark = ps.from_pandas(y_val)