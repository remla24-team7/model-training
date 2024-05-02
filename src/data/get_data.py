import urllib.request
import zipfile

URL = 'gdrive://48900e618890f4ce97201d1592d45045/dataset' #todo: fix repro get data from remote

EXTRACT_DIR = "../data"

x, y, z = urllib.request.urlretrieve(URL)
# with zipfile.ZipFile(zip_path, "r") as f:
#     f.extractall(EXTRACT_DIR)