import logging

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.sql import SparkSession

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


# Build the SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Income Model") \
    .config("spark.executor.memory", "1gb") \
    .getOrCreate()
# note: you might need to add export SPARK_LOCAL_IP=127.0.0.1

# Load in the data. For the sake of time, this dataset is extremely small.
# NOTE: In this case, the schema is being inferred. Most other times, you would specify your schema.
df = spark.read.csv("dataset.csv", header=True, inferSchema=True)
logging.info('Observing the raw data schema:')
df.printSchema()
logging.info('Observing a snippet of the raw data:')
df.show()
train_data, test_data = df.randomSplit([.8, .2], seed=1234)

#################
# Transformers
#################
CONT_COLS = ['age', 'fnlwgt', 'capital-gain', 'educational-num', 'capital-loss', 'hours-per-week']
CAT_COLS = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
            'native-country']
STR_IDX_COLS = ['{}_string_indexed'.format(x) for x in CAT_COLS]
HOTENCODED_COLS = ['{}_hot_encoded'.format(x) for x in CAT_COLS]
string_indexers = [StringIndexer(inputCol=cat_col, outputCol=str_idx_col).fit(df) for cat_col, str_idx_col in zip(CAT_COLS, STR_IDX_COLS)]
hot_encoded = OneHotEncoderEstimator(inputCols=STR_IDX_COLS, outputCols=HOTENCODED_COLS)
assembler = VectorAssembler(inputCols=HOTENCODED_COLS + CONT_COLS, outputCol="features")
income_string_idx = StringIndexer(inputCol="income", outputCol="income_str_idx")
base_pipeline_stages = string_indexers + [hot_encoded, assembler, income_string_idx]


#################
# Evaluator
#################
lr = LogisticRegression(labelCol="income_str_idx", featuresCol="features")

#################
# Build Pipeline and Train Model
#################

logistic_regression_pipeline = Pipeline(stages=base_pipeline_stages + [lr])
logistic_regression_pipeline_model = logistic_regression_pipeline.fit(train_data)
print("Coefficients: " + str(logistic_regression_pipeline_model.stages[-1].coefficients))
print("Intercept: " + str(logistic_regression_pipeline_model.stages[-1].intercept))

#################
# Predict With Model
#################
logistic_regression_predictions = logistic_regression_pipeline_model.transform(test_data)

#################
# Evaluate Model
#################
logistic_regression_predictions_selected = logistic_regression_predictions.select(CAT_COLS + CONT_COLS + ["income", "income_str_idx", "prediction", "probability"])
logistic_regression_predictions_selected.show(30)
logistic_regression_predictions_selected.groupby('income').agg({'income': 'count'}).show()
lr_pred = logistic_regression_predictions.select("income_str_idx", "prediction")
lr_accuracy_rate = lr_pred.filter(lr_pred.income_str_idx == lr_pred.prediction).count() / (lr_pred.count() * 1.0)
print('MODEL RESULTS:')
print("Overall Accuracy: {}".format(lr_accuracy_rate))

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='income_str_idx')
print('{}: {}'.format(evaluator.getMetricName(), evaluator.evaluate(logistic_regression_predictions)))


#################
# Save and Load Model
#################
logistic_regression_pipeline_model.write().overwrite().save('my_logistic_regression_model_2.model')
loaded_lr_model = PipelineModel.load("my_logistic_regression_model_2.model")
more_predictions = loaded_lr_model.transform(test_data)
print('\nLOADED MODEL RESULTS:')
print("Coefficients: " + str(loaded_lr_model.stages[-1].coefficients))
print("Intercept: " + str(loaded_lr_model.stages[-1].intercept))
lr_pred = more_predictions.select("income_str_idx", "prediction")
loaded_accuracy = lr_pred.filter(lr_pred.income_str_idx == lr_pred.prediction).count() / (lr_pred.count() * 1.0)
print("Overall Accuracy Loaded: {}".format(loaded_accuracy))