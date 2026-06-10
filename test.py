
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Example").getOrCreate()

data = [("Ravi", 30), ("Amit", 25)]
df = spark.createDataFrame(data, ["Name", "Age"])

df.show()
