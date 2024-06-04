import mlflow
from mlflow.models import infer_signature


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data, *args, **kwargs):
        # Set the tracking URI
    mlflow.set_tracking_uri("mlruns")

    # Create an experiment or set an existing one
    experiment_name = "ariel_lr"
    mlflow.set_experiment(experiment_name)

    #print(data[1])
    lr = data[1]
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(lr,
                                artifact_path="model"                                
                                )
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
     
    print(f"Logged model in experiment ID: {experiment_id}, run ID: {run_id}")

