import argparse
import json
import logging
import os
import distutils.util
from typing import Optional, List

from google.cloud import aiplatform


def trigger_pipeline_from_payload(payload: dict) -> aiplatform.PipelineJob:

    payload = convert_payload(payload)
    env = get_env()

    return trigger_pipeline(
        project_id=env["project_id"],
        location=env["location"],
        template_path=payload["attributes"]["template_path"],
        parameter_values=payload["data"],
        pipeline_root=env["pipeline_root"],
        service_account=env["service_account"],
        enable_caching=payload["attributes"]["enable_caching"],
    )


def trigger_pipeline(
    project_id: str,
    location: str,
    template_path: str,
    pipeline_root: str,
    service_account: str,
    parameter_values: dict = {},
    enable_caching: Optional[bool] = None,
) -> aiplatform.PipelineJob:
    # Initialise API client
    aiplatform.init(project=project_id, location=location)

    # Instantiate PipelineJob object
    pl = aiplatform.pipeline_jobs.PipelineJob(
        # Display name is required but seemingly not used
        # see
        # https://github.com/googleapis/python-aiplatform/blob/9dcf6fb0bc8144d819938a97edf4339fe6f2e1e6/google/cloud/aiplatform/pipeline_jobs.py#L260 # noqa
        display_name=template_path,
        enable_caching=enable_caching,
        template_path=template_path,
        parameter_values=parameter_values,
        pipeline_root=pipeline_root,
    )

    # Execute pipeline in Vertex
    pl.submit(
        service_account=service_account,
    )
    # pl.submit()

    return pl


def convert_payload(payload: dict) -> dict:
    """
    Processes the payload dictionary.
    Converts enable_caching and adds their defaults if they are missing.
    Args:
        payload (dict): Cloud Function event payload,
        or the contents of a payload JSON file
    """

    # make a copy of the payload so we are not modifying the original
    payload = payload.copy()

    # if payload["data"] is missing, add it as empty dict
    payload["data"] = payload.get("data", {})

    # if enable_caching value is in attributes, convert from str to bool
    # otherwise, it needs to be None
    if "enable_caching" in payload["attributes"]:
        payload["attributes"]["enable_caching"] = bool(
            distutils.util.strtobool(payload["attributes"]["enable_caching"])
        )
    else:
        payload["attributes"]["enable_caching"] = None



    return payload


def get_env() -> dict:
    """Get the necessary environment variables for pipeline runs,
    and return them as a dictionary.
    """

    project_id = os.environ["VERTEX_PROJECT_ID"]
    location = os.environ["VERTEX_LOCATION"]
    pipeline_root = os.environ["VERTEX_PIPELINE_ROOT"]
    service_account = os.environ["VERTEX_SA_EMAIL"]

    return {
        "project_id": project_id,
        "location": location,
        "pipeline_root": pipeline_root,
        "service_account": service_account,
    }

# python trigger.py --payload=./pipeline/config/config.json

def get_args(args: List[str] = None) -> argparse.Namespace:
    """Get args from command line args
    Args:
        event (dict): Event payload.
        context (google.cloud.functions.Context): Metadata for the event.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", help="Path to the config JSON file", type=str)
    return parser.parse_args(args)


def sandbox_run() -> aiplatform.PipelineJob:
    logging.basicConfig(level=logging.DEBUG)

    args = get_args()

    # Load JSON payload into a dictionary
    with open(args.payload, "r") as f:
        payload = json.load(f)

    return trigger_pipeline_from_payload(payload)


if __name__ == "__main__":
    sandbox_run()