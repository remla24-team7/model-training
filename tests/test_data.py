import pytest
import dvc.api
from lib_ml.dataset import load_dataset_file


@pytest.mark.parametrize("split", ("train", "val", "test"))
def test_data(split):
    dataset = dvc.api.params_show()["dirs"]["dataset"]
    urls, labels = load_dataset_file(f"{dataset}/{split}.txt")

    for url in urls:
        assert url, "URL should not be empty"

    for label in labels:
        assert label in ('legitimate', 'phishing'), "Label must be 'legitimate' or 'phishing'"
