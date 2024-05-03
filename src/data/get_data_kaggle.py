import os

# Set the credentials to use the API, These API keys will only be valid until I remake them for safety
os.environ['KAGGLE_USERNAME'] = "avilakathara"
os.environ['KAGGLE_KEY'] = "ab1c077561c76b993db8b9392613fd1d"

# TODO: Not a clean import, need to figure out the issue
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()

# Dataset to pull
dataset_name = "aravindhannamalai/dl-dataset"

# Directory to download into
download_dir = "data"

# Initialization code
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Download using API
api.dataset_download_files(dataset_name, path=download_dir, unzip=True)