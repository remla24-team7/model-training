"""
Test the data we are using, check if the labels are expected labels and urls not empty.

Can add other features as needed.
"""

import pytest


def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                data.append((parts[0], parts[1]))
    return data


# Test if expected labels
@pytest.mark.parametrize("label, url", read_data('../dataset/test.txt'))
def test_labels(label, url):
    assert label in ['legitimate', 'phishing'], "Label must be 'legitimate' or 'phishing'"


# Test if URLs are not empty
@pytest.mark.parametrize("label, url", read_data('../dataset/test.txt'))
def test_urls(label, url):
    assert url, "URL should not be empty"


@pytest.mark.parametrize("label, url", read_data('../dataset/train.txt'))
def test_labels(label, url):
    assert label in ['legitimate', 'phishing'], "Label must be 'legitimate' or 'phishing'"


# Test if URLs are not empty
@pytest.mark.parametrize("label, url", read_data('../dataset/train.txt'))
def test_urls(label, url):
    assert url, "URL should not be empty"


@pytest.mark.parametrize("label, url", read_data('../dataset/val.txt'))
def test_labels(label, url):
    assert label in ['legitimate', 'phishing'], "Label must be 'legitimate' or 'phishing'"


# Test if URLs are not empty
@pytest.mark.parametrize("label, url", read_data('../dataset/val.txt'))
def test_urls(label, url):
    assert url, "URL should not be empty"
