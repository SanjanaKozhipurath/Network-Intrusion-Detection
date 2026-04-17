import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from synapse.ml.isolationforest import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')


def train_models(df_pd, train_rf=True, train_lr=True, train_if=True,
                 progress_callback=None):

    results = []

    if progress_callback:
        progress_callback(0.05, "Initializing Spark session...")
    
    spark = SparkSession.builder \
        .appName("NSL-KDD IDS") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.11.2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    if progress_callback:
        progress_callback(0.10, "Converting to Spark DataFrame...")

    df = spark.createDataFrame(df_pd)
    df.cache()

    if progress_callback:
        progress_callback(0.15, "Splitting train/val/test sets...")

    train_df, val_df, test_df = df.randomSplit([0.70, 0.15, 0.15], seed=42)

    feature_cols = [c for c in df.columns if c != "label"]

    if progress_callback:
        progress_callback(0.20, "Assembling features...")

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    train_df = assembler.transform(train_df)
    val_df = assembler.transform(val_df)
    test_df = assembler.transform(test_df)

    train_df.cache()
    val_df.cache()
    test_df.cache()

    binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

    # ================= RF =================
    if train_rf:
        if progress_callback:
            progress_callback(0.25, "Training Random Forest...")
        
        start = time.time()

        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=10,
            maxDepth=8,
            seed=42
        )

        rf_model = rf.fit(train_df)
        
        if progress_callback:
            progress_callback(0.35, "Evaluating Random Forest...")
        
        test_pred = rf_model.transform(test_df)
        test_acc = acc_evaluator.evaluate(test_pred)

        prob_predictions = test_pred.select("label", "probability").toPandas()
        y_true = prob_predictions['label'].values
        y_prob = np.array([float(p[1]) for p in prob_predictions['probability']])
        roc_auc = roc_auc_score(y_true, y_prob)

        cm = confusion_matrix(
            test_pred.select("label").toPandas(),
            test_pred.select("prediction").toPandas()
        )

        results.append({
            'model_name': 'Random Forest',
            'model': rf_model,
            'training_time': time.time() - start,
            'val_accuracy': 0.0,
            'test_accuracy': test_acc,
            'roc_auc': roc_auc,
            'test_f1': 0.0,
            'confusion_matrix': cm,
            'feature_importances': rf_model.featureImportances.toArray()
        })

    # ================= LR =================
    if train_lr:
        if progress_callback:
            progress_callback(0.45, "Training Logistic Regression...")
        
        start = time.time()

        lr = LogisticRegression(
            labelCol="label",
            featuresCol="features",
            maxIter=100
        )

        lr_model = lr.fit(train_df)
        
        if progress_callback:
            progress_callback(0.55, "Evaluating Logistic Regression...")
        
        test_pred = lr_model.transform(test_df)
        test_acc = acc_evaluator.evaluate(test_pred)

        prob_predictions = test_pred.select("label", "probability").toPandas()
        y_true = prob_predictions['label'].values
        y_prob = np.array([float(p[1]) for p in prob_predictions['probability']])
        roc_auc = roc_auc_score(y_true, y_prob)

        cm = confusion_matrix(
            test_pred.select("label").toPandas(),
            test_pred.select("prediction").toPandas()
        )

        results.append({
            'model_name': 'Logistic Regression',
            'model': lr_model,
            'training_time': time.time() - start,
            'val_accuracy': 0.0,
            'test_accuracy': test_acc,
            'roc_auc': roc_auc,
            'test_f1': 0.0,
            'confusion_matrix': cm
        })

    # ================= IF + HYBRID =================
    if train_if:
        if progress_callback:
            progress_callback(0.65, "Preparing Isolation Forest...")

        from pyspark.ml.feature import StandardScaler

        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        scaler_model = scaler.fit(train_df)

        train_scaled = scaler_model.transform(train_df)
        test_scaled = scaler_model.transform(test_df)

        attack_ratio = df_pd['label'].mean()
        contamination_value = float(min(max(attack_ratio, 0.05), 0.4))

        iso = IsolationForest(
            featuresCol="scaledFeatures",
            predictionCol="anomaly",
            contamination=contamination_value
        )

        if progress_callback:
            progress_callback(0.70, "Training Isolation Forest...")

        start = time.time()

        iso_model = iso.fit(train_scaled)
        
        if progress_callback:
            progress_callback(0.80, "Evaluating Isolation Forest...")
        
        iso_pred = iso_model.transform(test_scaled)

        iso_time = time.time() - start

        iso_pd = iso_pred.select("anomaly", "label").toPandas()

        iso_pd['predicted_label'] = iso_pd['anomaly'].apply(
            lambda x: 1 if x == -1 else 0
        )

        iso_acc = np.mean(iso_pd['predicted_label'] == iso_pd['label'])

        if progress_callback:
            progress_callback(0.90, "Creating hybrid models...")

        # Hybrid RF
        if train_rf:
            rf_preds = rf_model.transform(test_df).select("prediction").toPandas()
            hybrid = [
                1 if iso_pd["anomaly"][i] == -1 else int(rf_preds["prediction"][i])
                for i in range(len(iso_pd))
            ]

            results.append({
                'model_name': 'Hybrid RF + IF',
                'training_time': iso_time,
                'val_accuracy': 0.0,
                'test_accuracy': np.mean(np.array(hybrid) == iso_pd['label']),
                'roc_auc': 0.0,
                'test_f1': 0.0
            })

        # Hybrid LR
        if train_lr:
            lr_preds = lr_model.transform(test_df).select("prediction").toPandas()
            hybrid = [
                1 if iso_pd["anomaly"][i] == -1 else int(lr_preds["prediction"][i])
                for i in range(len(iso_pd))
            ]

            results.append({
                'model_name': 'Hybrid LR + IF',
                'training_time': iso_time,
                'val_accuracy': 0.0,
                'test_accuracy': np.mean(np.array(hybrid) == iso_pd['label']),
                'roc_auc': 0.0,
                'test_f1': 0.0
            })

        # FIXED IF
        results.append({
            'model_name': 'Isolation Forest',
            'training_time': iso_time,
            'val_accuracy': 0.0,
            'test_accuracy': iso_acc,
            'roc_auc': 0.0,
            'test_f1': 0.0
        })

    if progress_callback:
        progress_callback(1.0, "Training complete!")

    spark.stop()
    return results
