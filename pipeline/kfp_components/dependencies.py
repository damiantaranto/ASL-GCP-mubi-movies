# ##################################################
# Required images for components + Vertex Training
# ##################################################

PYTHON37 = "europe-west1-docker.pkg.dev/vf-grp-aib-prd-mirror/generic-aib-images/python:3.7.12-buster"
TF_TRAINING_CONTAINER_IMAGE_URI = (
    "europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-6:latest"
)

# ###########################################
# Required packages + versions for components
# ###########################################

# Ensure that these versions are in sync with Pipfile

# TF specific
TENSORFLOW = "tensorflow==2.6.2"

# Google SDK specific
GOOGLE_CLOUD_BIGQUERY = "google-cloud-bigquery==2.30.0"
GOOGLE_CLOUD_STORAGE = "google-cloud-storage==1.42.2"
GOOGLE_CLOUD_AIPLATFORM = "google-cloud-aiplatform==1.10.0"

# Miscellaneous
APACHE_BEAM = "apache-beam==2.35.0"
PANDAS = "pandas==1.3.2"
