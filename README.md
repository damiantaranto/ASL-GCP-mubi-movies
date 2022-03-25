# ASL-GCP-mubi-movies
Google ASL 

Recommender system for customers for specific movie titles, giving the list of best customer.

## Configuration
Configurations can be found in `mubireco/config`, where `config.yaml.defaults` is its blueprint. Please find below a working example (please replace [PROJECT] placeholder and remove .defaults):
```
general:
  project: [PROJECT]
  region: us-west1-b
  model_name: mubi-reco
  bucket: mubi-kfp-artifact-store
  timestamp: 20220308175341

data_preprocessing:
  dataset_id: mubi_movie_data
  output_filename:
    train: train_mubi.tfdataset
    inference: inference_mubi.tfdataset
    movies: movies_mubi.tfdataset
```

## Data preprocessing
To run preprocessing step, just launch this command:
```
python3 mubireco/data_preprocessing.py --seq_length 10
```
In case `timestamp` is defined in the configuration, classes of data preprocessing are useful to load outputs:
```python
from mubireco.data.train import TrainDataset
from mubireco.utils.configuration import Configuration

config = Configuration("mubireco/config/config.yaml")
dataset = TrainDataset(config).read_tf_dataset().batch(10)

for batch in dataset.take(1):
    print(batch)
```
Output:
```
{'previous_movie_ratings': <tf.Tensor: shape=(2, 10), dtype=int32, numpy=
array([[ 1694, 15188, 41043, 17980, 39841, 30276, 30275, 36909, 38044,
          301],
       [ 3758,   825, 15931,  1596,  3672, 47112,  9894, 35011, 20834,
        26423]], dtype=int32)>, 'user_subscriber': <tf.Tensor: shape=(2,), dtype=int64, numpy=array([0, 0])>, 'previous_movie_years': <tf.Tensor: shape=(2, 10), dtype=float32, numpy=
array([[1960., 1975., 2011., 1973., 1998., 1989., 1983., 1987., 2004.,
        1972.],
       [1968., 1968., 1968., 1970., 1969., 1972., 1967., 2010., 1973.,
        1979.]], dtype=float32)>, 'user_trialist': <tf.Tensor: shape=(2,), dtype=int64, numpy=array([0, 0])>, 'user_eligible_for_trial': <tf.Tensor: shape=(2,), dtype=int64, numpy=array([1, 1])>, 'movie_id': <tf.Tensor: shape=(2,), dtype=int64, numpy=array([11431,  7827])>, 'user_has_payment_method': <tf.Tensor: shape=(2,), dtype=int64, numpy=array([0, 0])>}
```

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
