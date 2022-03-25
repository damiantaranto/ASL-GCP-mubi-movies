from kfp.v2 import compiler, dsl
from google_cloud_pipeline_components.experimental.custom_job.utils import (
    create_custom_training_job_op_from_component,
)

from pipeline.kfp_components.dependencies import TF_SERVING_CONTAINER_IMAGE_URI

import datetime

from google_cloud_pipeline_components.aiplatform import (
    EndpointCreateOp,
    ModelDeployOp,
    ModelUploadOp,
)

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

    movie_query = movies_dataset(
        project_id=project_id,
        data_root="{artifact}{time}/data".format(artifact=artifact_store, time=timestamp),
        movies_output_filename="movies_mubi.tfdataset"
    )

    train_query = train_dataset(
        project_id=project_id,
        data_root="{artifact}{time}/data".format(artifact=artifact_store, time=timestamp),
        movies_output_filename="train_mubi.tfdataset",
        seq_length=seq_length
    )

    inference_query = inference_dataset(
        project_id=project_id,
        data_root="{artifact}{time}/data".format(artifact=artifact_store, time=timestamp),
        movies_output_filename="inference_mubi.tfdataset",
        seq_length=seq_length
    )

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
            .after(movie_query, train_query, inference_query)
            .set_display_name("Vertex Training for TF model")
    )

    model_upload_task = ModelUploadOp(
        project=project_id,
        display_name=f"{model_name}-kfp-model-upload-job",
        artifact_uri=f"{artifact_store}20220311195134/20220317091043/models",
        serving_container_image_uri=TF_SERVING_CONTAINER_IMAGE_URI,
    ).after(train_model)

    create_endpoint_op = EndpointCreateOp(
        project=project_id,
        display_name=f"{model_name}-kfp-create-endpoint-job",
    ).after(model_upload_task)

    model_deploy_op = ModelDeployOp(
        model=model_upload_task.outputs["model"],
        endpoint=create_endpoint_op.outputs['endpoint'],
        dedicated_resources_machine_type="n1-standard-16",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    ).after(create_endpoint_op)

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
        machine_type="n1-standard-64",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=4
    )
    compile()
