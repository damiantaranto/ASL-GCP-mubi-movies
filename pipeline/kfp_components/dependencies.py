# ##################################################
# Required images for components + Vertex Training
# ##################################################

PYTHON37 = "python:3.8"
TF_TRAINING_CONTAINER_IMAGE_URI = (
    "europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-6:latest"
)

TF_SERVING_CONTAINER_IMAGE_URI = (
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest"
)

# ###########################################
# Required packages + versions for components
# ###########################################

# Ensure that these versions are in sync with Pipfile

# TF specific
TENSORFLOW = "tensorflow==2.6.2"
TENSORFLOW_RECOMMENDERS = "tensorflow-recommenders==0.6.0"

# Google SDK specific
GOOGLE_CLOUD_BIGQUERY = "google-cloud-bigquery==2.34.2"
GOOGLE_CLOUD_STORAGE = "google-cloud-storage==1.42.2"
GOOGLE_CLOUD_AIPLATFORM = "google-cloud-aiplatform==1.10.0"

# Miscellaneous
APACHE_BEAM = "apache-beam==2.35.0"
PANDAS = "pandas==1.3.2"
PYARROW = "pyarrow==7.0.0"
