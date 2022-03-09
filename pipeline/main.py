import json
import pathlib

from kfp.v2 import compiler, dsl
from google_cloud_pipeline_components.experimental.custom_job.utils import (
    create_custom_training_job_op_from_component,
)
from pipeline import generate_query

from pipeline.kfp_components.tensorflow import train_tensorflow_model

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
    ingestion_dataset_id: str = ""
):
    # Create variables to ensure the same arguments are passed
    # into different components of the pipeline
    ingestion_table = "taxi_trips"
    table_suffix = "_tf"  # suffix to table names
    ingested_table = "ingested_data" + table_suffix

    queries_folder = pathlib.Path(__file__).parent / "queries"

    ingest_query = generate_query(
        queries_folder / "ingest.sql",
        source_dataset=f"{ingestion_project_id}"
    )


def compile():
    """
    Uses the kfp compiler package to compile the pipeline function into a workflow yaml
    Args:
        None
    Returns:
        None
    """
    compiler.Compiler().compile(
        pipeline_func=tensorflow_pipeline,
        package_path="training.json",
        type_check=False,
    )

if __name__ == "__main__":
    custom_train_job = create_custom_training_job_op_from_component(
        component_spec=train_tensorflow_model,
        replica_count=1,
        machine_type="n1-standard-4",
    )
    compile()