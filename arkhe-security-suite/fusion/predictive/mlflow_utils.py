import mlflow

def log_experiment(params, metrics, version=None):
    if version:
        mlflow.set_tag("version", version)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

def create_experiment(name):
    try:
        return mlflow.create_experiment(name)
    except:
        return mlflow.get_experiment_by_name(name).experiment_id
