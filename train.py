# --- Libraries
import copy
import time

import torch
from icecream import ic

from helpers import get_device, one_hot_embedding
from losses import relu_evidence
from icecream import ic
# ---

# --- Global variables
DEBUG = False
# ---


def train_model(
    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    uncertainty=False,
):
    since = time.time()  # Start time

    if not device:
        device = get_device()

    # --- Best model weights and accuracy variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # ---

    # --- Variables for storing losses, accuracy and evidence
    losses = {"loss": [], "phase": [], "epoch": []}
    accuracies = {"accuracy": [], "phase": [], "epoch": []}
    # evidences = {"evidence": [], "type": [], "epoch": []}
    # ---

    # --- Epoch loop
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            correct = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # --- Move input and label tensors to the specified device
                inputs = inputs.to(device)  # shape: [b, 1, 28, 28]
                labels = labels.to(device)  # shape: [b]
                # ---

                # ic(inputs.shape)
                # ic(labels.shape)
                # input("Press 'ENTER' to continue...")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:
                        # Example: [5, ...] -> [(0, 0, 0, 0, 0, 1, 0, 0, 0, 0), ...]
                        y = one_hot_embedding(labels, num_classes)
                        y = y.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device
                        )

                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                        acc = torch.mean(match)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                        total_evidence = torch.sum(evidence, 1, keepdim=True)
                        mean_evidence = torch.mean(total_evidence)
                        mean_evidence_succ = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match
                        ) / torch.sum(match + 1e-20)
                        mean_evidence_fail = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * (1 - match)
                        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        # Compute Loss (nn.CrossEntropyLoss())
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            losses["loss"].append(epoch_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracies["accuracy"].append(epoch_acc.item())
            accuracies["phase"].append(phase)
            accuracies["epoch"].append(epoch)

            print(
                "{} loss: {:.4f} acc: {:.4f}".format(
                    phase.capitalize(), epoch_loss, epoch_acc
                )
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracies)

    return model, metrics
