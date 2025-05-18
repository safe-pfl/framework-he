from utils.log import Log
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
import torch


def train(
    model: Module, loader: DataLoader, optimizer, epochs, device: str, log: "Log"
):
    criterion = CrossEntropyLoss().to(device, non_blocking=True)
    model.train()
    model.float()  # Ensure model is in float32 precision

    running_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            # Convert input to float32 precision
            images = images.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels.long())

            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        if epoch > 1:
            log.info(f"[{epoch + 1}] loss: {running_loss / len(loader):.3f}")

    return model, running_loss / len(loader)
