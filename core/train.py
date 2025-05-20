from utils.log import Log
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
import torch


def train(
    model: Module,
    loader: DataLoader,
    optimizer,
    epochs,
    device: str,
    log: "Log",
    track_gradients=False,
):
    criterion = CrossEntropyLoss().to(device, non_blocking=True)
    model.train()
    model.float()  # Ensure model is in float32 precision

    running_loss = 0.0

    # For tracking gradients if needed
    accumulated_grads = None
    if track_gradients:
        accumulated_grads = []
        for param in model.parameters():
            if param.requires_grad:
                accumulated_grads.append(torch.zeros_like(param, device=device))
            else:
                accumulated_grads.append(None)

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

            # Track gradients if needed
            if track_gradients:
                for i, param in enumerate(model.parameters()):
                    if param.requires_grad and param.grad is not None:
                        accumulated_grads[i] += param.grad.detach().abs()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        if epoch > 1:
            log.info(f"[{epoch + 1}] loss: {running_loss / len(loader):.3f}")

    if track_gradients:
        return model, running_loss / len(loader), accumulated_grads
    else:
        return model, running_loss / len(loader)
