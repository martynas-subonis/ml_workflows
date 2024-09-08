from kfp.dsl import ClassificationMetrics, Dataset, Input, Metrics, Model, Output


def eval_model(
    data_bucket: str,
    test_split_info: Input[Dataset],
    torch_model: Input[Model],
    eval_metrics: Output[Metrics],
    confusion_matrix_output: Output[ClassificationMetrics],
) -> None:
    import logging
    import time
    from json import load
    from pathlib import Path
    from typing import Any

    import numpy as np
    import torch
    from google.cloud.storage import Client, transfer_manager
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import (
        MobileNet_V3_Small_Weights,
        mobilenet_v3_small,
    )

    logging.info("Started evaluate model task.")
    start_time = time.time()
    client = Client()
    bucket = client.bucket(data_bucket)
    with open(test_split_info.path) as f:
        test_split = load(f)

    logging.info("Downloading test_images...")
    transfer_manager.download_many_to_path(bucket, test_split, destination_directory="test_images", max_workers=8, skip_if_exists=True)

    model = mobilenet_v3_small()
    num_classes = sum(1 for item in Path("test_images").iterdir() if item.is_dir())
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    model.load_state_dict(torch.load(torch_model.path))

    # MPS is considered in case one wants to run this code outside Kubeflow pipelines.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model.to(device)
    model.eval()

    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    test_dataset = ImageFolder("test_images", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    preds_acc: list[np.int64] = []
    labels_acc: list[np.int64] = []

    with torch.no_grad():
        logging.info("Running predictions for test images...")
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            preds_acc.extend(preds.cpu().numpy())
            labels_acc.extend(labels.cpu().numpy())

    all_preds: np.ndarray[Any, np.dtype[np.int64]] = np.array(preds_acc)
    all_labels: np.ndarray[Any, np.dtype[np.int64]] = np.array(labels_acc)
    class_names = [class_name for class_name, _ in test_dataset.class_to_idx.items()]

    confusion_matrix_output.log_confusion_matrix(class_names, confusion_matrix(all_labels, all_preds).tolist())

    for name, metric_fn in (
        ("weightedPrecision", precision_score),
        ("weightedRecall", recall_score),
        ("weightedF1Score", f1_score),
    ):
        eval_metrics.log_metric(name, metric_fn(all_labels, all_preds, average="weighted"))

    eval_metrics.log_metric("timeTakenSeconds", round(time.time() - start_time, 2))
    logging.info("Successfully finished weather model evaluation.")
