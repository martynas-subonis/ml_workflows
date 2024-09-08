from kfp.dsl import Dataset, Metrics, Output


def prep_data(
    data_bucket: str,
    random_seed: int,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
    train_split_info: Output[Dataset],
    val_split_info: Output[Dataset],
    test_split_info: Output[Dataset],
    data_prep_metrics: Output[Metrics],
) -> None:
    import logging
    import time
    from collections import Counter
    from json import dump

    from google.cloud.storage import Client
    from sklearn.model_selection import train_test_split

    start_time = time.time()
    logging.info("Started data preparation task.")

    for val in (train_ratio, test_ratio, val_ratio):
        if val <= 0 or val >= 1:
            raise ValueError("Train, test and validation ratios must be in range (0, 1).")

    if train_ratio + test_ratio + val_ratio != 1.0:
        raise ValueError("Train, test and validation ratios must sum-up to 1.")

    client = Client()
    bucket = client.bucket(data_bucket)
    img_paths: list[str] = [blob.name for blob in client.list_blobs(bucket)]

    def derive_class(img_path: str) -> str:
        return img_path.split("/")[0]

    classes = [derive_class(path) for path in img_paths]
    x_train, x_val_test = train_test_split(img_paths, stratify=classes, test_size=1 - train_ratio, random_state=random_seed)

    val_test_classes = [derive_class(path) for path in x_val_test]
    x_val, x_test = train_test_split(
        x_val_test, stratify=val_test_classes, test_size=test_ratio / (test_ratio + val_ratio), random_state=random_seed
    )

    for name, data_set, artifact in (
        ("train", x_train, train_split_info),
        ("validation", x_val, val_split_info),
        ("test", x_test, test_split_info),
    ):
        data_prep_metrics.log_metric(f"{name}Size", len(data_set))
        counts = Counter(map(derive_class, data_set))
        for key, value in counts.items():
            data_prep_metrics.log_metric(f"{name}{key.capitalize()}", value)
        with open(artifact.path, "w") as f:
            dump(data_set, f)

    data_prep_metrics.log_metric("timeTakenSeconds", round(time.time() - start_time, 2))
    logging.info("Successfully finished data preparation task.")
