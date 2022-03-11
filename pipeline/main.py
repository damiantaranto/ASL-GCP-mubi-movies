from kfp.v2 import compiler, dsl
from google_cloud_pipeline_components.experimental.custom_job.utils import (
    create_custom_training_job_op_from_component,
)

from pipeline.kfp_components.preprocessing.movies import movies_dataset


@dsl.pipeline(name="tensorflow-train-pipeline")
def tensorflow_pipeline(
        project_id: str,
        timestamp: str,
        model_name: str,
):
    # Create variables to ensure the same arguments are passed
    # into different components of the pipeline

    artifact_store = "gs://{model}-kfp-artifact-store".format(model=model_name)

    movie_query = movies_dataset(
        project_id=project_id,
        data_root="{artifact}/{time}/data".format(artifact=artifact_store, time=timestamp),
        movies_output_filename="movies_mubi.tfdataset"
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
    # custom_train_job = create_custom_training_job_op_from_component(
    #     component_spec=tensorflow_pipeline,
    #     replica_count=1,
    #     machine_type="n1-standard-4",
    # )
    compile()
