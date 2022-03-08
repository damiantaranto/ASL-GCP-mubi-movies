import os
import tensorflow as tf

from mubireco.data.base import DataPrep


class MoviesDataset(DataPrep):
    """Movies dictionary: TF dataset preparation"""

    name_transformation = "movies"

    query = f"""SELECT DISTINCT
        ratings.movie_id,
        movies.movie_title,
    FROM `__DATASET_ID__.mubi_ratings_data` ratings
    JOIN `__DATASET_ID__.mubi_movie_data` movies ON
        ratings.movie_id = movies.movie_id"""

    def __init__(self, configuration, **kwargs):
        super().__init__(configuration, **kwargs)
        self._start_frame = None
        self._end_frame = None

    def set_data_path(self) -> str:
        return os.path.join(self.data_root, self.name_transformation)

    def set_output_filename(self) -> str:
        return self.output_filename.get(self.name_transformation)

    @staticmethod
    def get_features_dict(rows) -> dict:
        dict_features = dict(
            **rows[["movie_id"]].astype("int"),
            **rows[["movie_title"]].astype("str"),
        )
        return dict_features

    def run(self):
        try:
            assert not tf.io.gfile.exists(self.output_path)
        except AssertionError:
            raise ValueError("Dataset already exists, load it. Remove timestamp from config in case of new run")

        rows = self.load_dataset(
            self.query
            .replace("__DATASET_ID__", str(self.dataset_id))
        )

        dict_features = self.get_features_dict(rows)
        del rows

        self.save_tf_dataset(dict_features)
