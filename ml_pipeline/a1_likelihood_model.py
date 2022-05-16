<<<<<<< HEAD
'''Used for first ML model in our pipeline.'''
=======
X_train_spark = spark.createDataFrame(X_train)
X_val_spark = spark.createDataFrame(X_val)
y_train_spark = ps.from_pandas(y_train)
y_val_spark = ps.from_pandas(y_val)
>>>>>>> e71cd00eaf081f58f0d6bc90abf0653ee5c8f4ef
