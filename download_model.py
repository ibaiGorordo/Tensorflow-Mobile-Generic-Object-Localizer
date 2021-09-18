import requests
import tarfile

url = "https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1?tf-hub-format=compressed"
response = requests.get(url, stream=True)
file = tarfile.open(fileobj=response.raw, mode="r|gz")
file.extractall(path="models/saved_model")