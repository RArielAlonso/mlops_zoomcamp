import mlflow

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def log_model(X_train,lr):
    with mlflow.start_run():
        signature = mlflow.infer_signature(X_train, lr.predict(X_train))

        mlflow.sklearn.log_model(lr,
                            artifact_path="model",
                            signature=signature
                            )
