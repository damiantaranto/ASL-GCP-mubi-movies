# ASL-GCP-mubi-movies
Google ASL 

To run the pipeline - first set the following as you environemt varaibles - also in ```env.sh```
```
export PIPELINE_FILES_GCS_PATH=gs://movie_pipeline_staging/pipelines
export VERTEX_LOCATION=us-central1
export VERTEX_PIPELINE_ROOT=gs://movie_pipeline_vertex_root
export VERTEX_PROJECT_ID=qwiklabs-gcp-04-424f1fdacc59
export VERTEX_SA_EMAIL=qwiklabs-gcp-04-424f1fdacc59@qwiklabs-gcp-04-424f1fdacc59.iam.gserviceaccount.com
```
Then you need to compile the pipeline
```
python3 -m pipeline.main
```
This will create the training.json file for vertex
To run the pipeline, run the following
```
python3 trigger.py --payload=./pipeline/config/config.json
```
