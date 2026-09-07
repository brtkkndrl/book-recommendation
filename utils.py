import kagglehub
import os

def get_dataset(url):
    path = kagglehub.dataset_download(url)

    print("Path to dataset files:", path)

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.relpath(os.path.join(dirname, filename), path))

    return path