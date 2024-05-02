import urllib.request
import zipfile

URL = 'https://surfdrive.surf.nl/files/index.php/s/OZRd9BcxhGkxTuy/download' # v2, 2000 datapoints

EXTRACT_DIR = "smsspamcollection"

zip_path, _ = urllib.request.urlretrieve(URL)
with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(EXTRACT_DIR)