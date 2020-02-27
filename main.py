import logging

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.sql import SparkSession

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


class CensusModel(object):
    CONT_COLS = ['age', 'fnlwgt', 'capital-gain', 'educational-num', 'capital-loss', 'hours-per-week']
    CAT_COLS = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
                'native-country']

    def __init__(self, df):
        self.df = df

    def _build_transformers(self):
        STR_IDX_COLS = ['{}_string_indexed'.format(x) for x in self.CAT_COLS]
        HOTENCODED_COLS = ['{}_hot_encoded'.format(x) for x in self.CAT_COLS]
        string_indexers = [StringIndexer(inputCol=cat_col, outputCol=str_idx_col).fit(self.df) for cat_col, str_idx_col in zip(self.CAT_COLS, STR_IDX_COLS)]
        hot_encoded = OneHotEncoderEstimator(inputCols=STR_IDX_COLS, outputCols=HOTENCODED_COLS)
        assembler = VectorAssembler(inputCols=HOTENCODED_COLS + self.CONT_COLS, outputCol="features")
        income_string_idx = StringIndexer(inputCol="income", outputCol="income_str_idx")
        return string_indexers + [hot_encoded, assembler, income_string_idx]

    def build_pipeline_single_estimator(self, model_estimator):
        return Pipeline(stages=self._build_transformers() + [model_estimator])

    @staticmethod
    def train_model(train_data, pipeline):
        pipeline_model = pipeline.fit(train_data)
        return pipeline_model

    def fit_model(self, test_data, pipeline_model, show_snippet=True):
        results = pipeline_model.transform(test_data)
        if show_snippet:
            relevant_results = results.select(self.CAT_COLS + self.CONT_COLS + ["income", "income_str_idx", "prediction", "probability"])
            relevant_results.show(30)
        return results

    def create_test_and_train(self, ratio=(.8, .2), seed=1234):
        return self.df.randomSplit(ratio, seed=seed)

    @staticmethod
    def evaluate_model(results, evaluator):
        accuracy_rate = results.filter(results.income_str_idx == results.prediction).count() / (results.count() * 1.0)
        print("Overall Accuracy: {}".format(accuracy_rate))
        print('{}: {}'.format(evaluator.getMetricName(), evaluator.evaluate(results)))


def main():
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

    census_model = CensusModel(df)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='income_str_idx')
    training_set, test_set = census_model.create_test_and_train()

    # logistic regression
    logger.info('LOGISTIC REGRESSION')
    lr = LogisticRegression(labelCol="income_str_idx", featuresCol="features")
    lr_pipeline = census_model.build_pipeline_single_estimator(lr)
    lr_model = census_model.train_model(training_set, lr_pipeline)
    lr_predictions = census_model.fit_model(test_set, lr_model)
    census_model.evaluate_model(lr_predictions, evaluator)

    # random forest
    logger.info('RANDOM FOREST')
    rf = RandomForestClassifier(labelCol="income_str_idx", featuresCol="features")
    rf_pipeline = census_model.build_pipeline_single_estimator(rf)
    rf_model = census_model.train_model(training_set, rf_pipeline)
    rf_predictions = census_model.fit_model(test_set, rf_model)
    census_model.evaluate_model(rf_predictions, evaluator)

    # comparing
    print('\nLOGISTIC REGRESSION RESULTS')
    census_model.evaluate_model(lr_predictions, evaluator)
    print('\nRANDOM FOREST RESULTS')
    census_model.evaluate_model(rf_predictions, evaluator)

    # save and load
    lr_model.write().overwrite().save('my_logistic_regression_model.model')
    rf_model.write().overwrite().save('my_random_forest_model.model')
    lr_model_loaded = PipelineModel.load("my_logistic_regression_model.model")
    rf_model_loaded = PipelineModel.load("my_random_forest_model.model")
    # du - hd1
    lr_predictions_loaded = census_model.fit_model(test_set, lr_model_loaded, show_snippet=False)
    rf_predictions_loaded = census_model.fit_model(test_set, rf_model_loaded, show_snippet=False)
    print('\nLOADED MODEL LOGISTIC REGRESSION RESULTS')
    census_model.evaluate_model(lr_predictions_loaded, evaluator)
    print('\nLOADED MODEL RANDOM FOREST RESULTS')
    census_model.evaluate_model(rf_predictions_loaded, evaluator)


if __name__ == '__main__':
    main()