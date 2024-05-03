from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

# Create a SparkSession
spark = SparkSession.builder.appName('logreg').getOrCreate()

# Load training data
training = spark.read.format("libsvm").load("sample_libsvm_data.txt")

# Initialize the logistic regression model
lr = LogisticRegression()

# Fit the model to the data
lrModel = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# Stop the SparkSession
spark.stop()