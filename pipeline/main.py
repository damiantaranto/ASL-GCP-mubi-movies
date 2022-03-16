from kfp.v2 import compiler, dsl
from google_cloud_pipeline_components.experimental.custom_job.utils import (
    create_custom_training_job_op_from_component,
)

import datetime

from pipeline.kfp_components.preprocessing.movies import movies_dataset
from pipeline.kfp_components.preprocessing.train_dataset import train_dataset
from pipeline.kfp_components.preprocessing.inference_dataset import inference_dataset

from pipeline.kfp_components.tensorflow.train import train_tensorflow_model

@dsl.pipeline(name="tensorflow-train-pipeline")
def tensorflow_pipeline(
        project_id: str,
        model_name: str,
        seq_length: int,
):
    # Create variables to ensure the same arguments are passed
    # into different components of the pipeline

    artifact_store = "gs://{model}-kfp-artifact-store/".format(model=model_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # movie_query = movies_dataset(
    #     project_id=project_id,
    #     data_root="{artifact}{time}/data".format(artifact=artifact_store, time=timestamp),
    #     movies_output_filename="movies_mubi.tfdataset"
    # )
    #
    # train_query = train_dataset(
    #     project_id=project_id,
    #     data_root="{artifact}{time}/data".format(artifact=artifact_store, time=timestamp),
    #     movies_output_filename="train_mubi.tfdataset",
    #     seq_length=seq_length
    # )

    # val_query = val_dataset(
    #     project_id=project_id,
    #     data_root="{artifact}{time}/data".format(artifact=artifact_store, time=timestamp),
    #     val_output_filename="val_mubi.tfdataset",
    #     seq_length=seq_length
    # )

    # inference_query = inference_dataset(
    #     project_id=project_id,
    #     data_root="{artifact}{time}/data".format(artifact=artifact_store, time=timestamp),
    #     movies_output_filename="inference_mubi.tfdataset",
    #     seq_length=seq_length
    # )

    # train_model = train_tensorflow_model(
    #     data_root="{artifact}{time}/model".format(artifact=artifact_store, time=timestamp),
    #     movies_output_filename="movies_mubi.tfdataset",
    #     train_output_filename="train_mubi.tfdataset",
    #     inference_output_filename="inference_mubi.tfdataset",
    #     timestamp=timestamp,
    #     artifact_store=artifact_store,
    #     # # Training wrapper specific parameters
    #     # project=project_id,
    #     # location="us-central1",
    # ).after(inference_query, movie_query, train_query)

    train_model = (
        custom_train_job(
            data_root="{artifact}{time}/data".format(artifact=artifact_store, time=timestamp),
            movies_output_filename="movies_mubi.tfdataset",
            train_output_filename="train_mubi.tfdataset",
            inference_output_filename="inference_mubi.tfdataset",
            artifact_store=artifact_store,
            timestamp=timestamp,
            bucket_name="{model}-kfp-artifact-store".format(model=model_name),
            # Training wrapper specific parameters
            project=project_id,
            location="us-central1",
        )
            # .after(inference_query, movie_query, train_query)
            .set_display_name("Vertex Training for TF model")
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
        machine_type="n1-standard-32",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=2
    )
    compile()
