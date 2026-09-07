import kagglehub

def get_datasets():
    # Download latest version
    path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")

    print("Path to dataset files:", path)


def main() -> None:
    print("Hello from book-recommendation!")
