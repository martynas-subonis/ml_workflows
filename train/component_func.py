from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output


def train_model(
    data_bucket: str,
    random_seed: int,
    epochs: int,
    train_split_info: Input[Dataset],
    val_split_info: Input[Dataset],
    train_metrics: Output[Metrics],
    loss_plot: Output[Artifact],
    torch_model: Output[Model],
    onnx_model: Output[Model],
    onnx_with_transform_model: Output[Model],
) -> None:
    import logging
    import os
    import random
    import time
    from json import load
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import onnx
    import torch
    import torch.nn.functional as F
    import torchvision
    from google.cloud.storage import Client, transfer_manager
    from torch.nn import Module
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import (
        MobileNet_V3_Small_Weights,
        MobileNetV3,
        mobilenet_v3_small,
    )
    from torchvision.transforms.functional import pad

    logging.info("Started train model task.")
    # Reproducibility:
    # https://pytorch.org/docs/2.3/notes/randomness.html#reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    start_time = time.time()
    client = Client()
    bucket = client.bucket(data_bucket)
    for split_info, dest_dir in ((train_split_info.path, "train_images"), (val_split_info.path, "val_images")):
        with open(split_info) as f:
            split = load(f)

        logging.info(f"Downloading {dest_dir}...")
        transfer_manager.download_many_to_path(bucket, split, destination_directory=dest_dir, max_workers=8, skip_if_exists=True)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # MPS is considered in case one wants to run this code outside Kubeflow pipelines.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Fine-tuning part - freezing all layers.
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tuning part - replacing classifier head for our specific problem domain.
    num_classes = sum(1 for item in Path("train_images").iterdir() if item.is_dir())
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.classifier[-1].parameters(), momentum=0.9, weight_decay=1e-2)

    transform = weights.transforms()
    train_dataset, val_dataset = ImageFolder("train_images", transform=transform), ImageFolder("val_images", transform=transform)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=32, shuffle=True), DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_losses, val_losses, train_acc, val_acc = [], [], torch.tensor(0).to(device), torch.tensor(0).to(device)

    for epoch in range(epochs):
        model.train()
        run_loss, run_correct = 0.0, torch.tensor(0).to(device)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            run_loss += loss.item() * inputs.size(0)
            run_correct += torch.sum(preds == labels.data)

        train_loss = run_loss / len(train_dataset)
        train_losses.append(train_loss)
        train_acc = run_correct.float() / len(train_dataset)

        model.eval()
        run_loss, run_correct = 0.0, torch.tensor(0).to(device)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                run_loss += loss.item() * inputs.size(0)
                run_correct += torch.sum(preds == labels.data)

        val_loss = run_loss / len(val_dataset)
        val_losses.append(val_loss)
        val_acc = run_correct.float() / len(val_dataset)

        logging.info(f"Epoch: {epoch + 1}. Total number of epochs: {epochs}.")
        logging.info(f"Training loss: {train_loss}, training accuracy: {train_acc}.")
        logging.info(f"Validation loss: {val_loss}, validation accuracy: {val_acc}.")

    plot_epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(plot_epochs, train_losses, "b-", label="Training Loss")
    plt.plot(plot_epochs, val_losses, "r-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot.path)
    plt.close()

    train_metrics.log_metric("trainAccuracy", train_acc.item())
    train_metrics.log_metric("valAccuracy", val_acc.item())
    if train_losses and val_losses:
        train_metrics.log_metric("trainLoss", train_losses[-1])
        train_metrics.log_metric("valLoss", val_losses[-1])

    setup_info = (
        f"torch-{torch.__version__}, torchvision-{torchvision.__version__}, "
        f"numpy-{np.__version__}, model={mobilenet_v3_small.__name__}, "
        f"weights={MobileNet_V3_Small_Weights.__name__}, {random_seed=}, {epochs=}"
    )
    torch.save(model.state_dict(), torch_model.path)
    torch_model.framework = setup_info
    model_input = torch.randn(1, 3, transform.crop_size[0], transform.crop_size[0]).to(device)
    opset_version = 17
    torch.onnx.export(model, model_input, f"{onnx_model.path}.onnx", opset_version=opset_version)
    onnx_model.framework = f"{setup_info}, onnx-{onnx.__version__}, {opset_version=}"

    class ModelWithTransforms(Module):  # type: ignore[misc]
        def __init__(self, model: MobileNetV3) -> None:
            super(ModelWithTransforms, self).__init__()
            self.model = model
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.register_buffer("targ_h", torch.tensor(224))
            self.register_buffer("targ_w", torch.tensor(224))

        def transform(self, img: torch.Tensor) -> torch.Tensor:
            resized = F.interpolate(img, size=256, mode="bilinear", align_corners=False)
            _, _, curr_h, curr_w = resized.shape
            pad_h = torch.clamp(self.targ_h - curr_h, min=0)
            pad_w = torch.clamp(self.targ_w - curr_w, min=0)
            padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            padded = pad(resized, padding).to(device)
            start_h = torch.clamp((curr_h + pad_h - self.targ_h) // 2, min=0)
            start_w = torch.clamp((curr_w + pad_w - self.targ_w) // 2, min=0)
            cropped = padded[..., start_h : start_h + self.targ_h, start_w : start_w + self.targ_w]
            normalized = (cropped - self.mean.to(device)) / self.std.to(device)
            return normalized

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.transform(x)
            return self.model(x)

    model_with_transform = ModelWithTransforms(model)
    model_with_transform.to(device)
    torch.onnx.export(
        model_with_transform,
        model_input,
        f"{onnx_with_transform_model.path}.onnx",
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},  # Dynamic batch size, height, and width
            "output": {0: "batch_size"},  # Dynamic batch size for output
        },
    )
    onnx_with_transform_model.framework = f"{setup_info}, onnx-{onnx.__version__}, {opset_version=}"

    train_metrics.log_metric("timeTakenSeconds", round(time.time() - start_time, 2))
    logging.info("Successfully finished weather model training.")
