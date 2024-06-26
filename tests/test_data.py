# pylint: disable=no-name-in-module,import-error
import pytest
from lib_ml.dataset import load_dataset_file


@pytest.mark.parametrize("split", ("train", "val", "test"))
def test_data(params, split):
    dataset_dir = params["dirs"]["dataset"]
    urls, labels = load_dataset_file(f"{dataset_dir}/{split}.txt")

    for url in urls:
        assert url, "URL should not be empty"

    for label in labels:
        assert label in ('legitimate', 'phishing'), "Label must be 'legitimate' or 'phishing'"
