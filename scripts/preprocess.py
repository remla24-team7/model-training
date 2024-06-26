import dvc.api
# pylint: disable=import-error
from lib_ml.dataset import preprocess_dataset

params = dvc.api.params_show()

preprocess_dataset(
    dataset_dir=params["dirs"]["dataset"],
    outputs_dir=params["dirs"]["outputs"]["preprocess"],
    sequence_length=params["preprocess"]["sequence_length"],
)
