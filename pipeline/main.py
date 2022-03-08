from kfp.v2 import compiler, dsl

@dsl.pipeline(name="tensorflow-train-pipeline")
def tensorflow_pipeline(
    project_id: str,
    project_location: str,
    ingestion_project_id: str,
    tfdv_schema_path: str,
    tfdv_train_stats_path: str,
    train_script_path: str,
    model_name: str,
    scheduled_time: str,
    dataset_id: str = "preprocessing",
    dataset_location: str = "US",
):