# --- Libraries
import copy
import time

import torch
from icecream import ic  # Noqa

from helpers import get_device, one_hot_encoding
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
    annealing_step,
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
    # ---

    # --- Epoch loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
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

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # --- Move input and label tensors to the specified device
                inputs = inputs.to(device)  # shape: [b, 1, 28, 28]
                labels = labels.to(device)  # shape: [b]
                # ---

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:
                        # One-hot encoding example : [5, ...] -> [(0, 0, 0, 0, 0, 1, 0, 0, 0, 0), ...]
                        y = one_hot_encoding(labels, num_classes)
                        y = y.to(device)
                        outputs = model(inputs)  # Logits?, shape: [b, 10]

                        """
                        Usually, the softmax should be applied now, but in Evidential Deep Learning, this last operation
                        is replaced by the relu_evidence() function. However, can we just apply the argmax function on 
                        the logits to obtain the predicted class, which is then used to calculate the accuracy?
                        """

                        _, preds = torch.max(outputs, 1)  # Predicted class?, shape: [b]

                        # TODO: Revisei até aqui

                        # Compute Evidential Loss (edl_digamma_loss, edl_log_loss, or edl_mse_loss)
                        loss = criterion(outputs, y.float(), epoch, num_classes, annealing_step, device)

                        # --- Compute Accuracy
                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))  # shape: [b] -> [b, 1]
                        acc = torch.mean(match)  # Batch accuracy, average of matches

                        # --- Compute the Evidence vector and the Dirichlet parameters
                        evidence = relu_evidence(outputs)  # Evidence vector, f(x_i, theta)
                        alpha = evidence + 1  # Dirichlet parameters, alpha_i = f(x_i, theta) + 1

                        # --- Compute the uncertainty (u) of the prediction, which is the number of classes (K) divided
                        # by the Dirichlet strength (S)
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)  # K / sum(alpha_i)
                        # ---

                        if DEBUG:
                            ic(inputs.shape)
                            ic(labels.shape)
                            ic(labels[0])
                            ic(y[0])
                            ic(y[0].float())
                            ic(outputs[0])
                            ic(preds[0])
                            ic(loss)
                            # ic(torch.eq(preds, labels))  # Bool
                            # ic(torch.eq(preds, labels).float())  # Float
                            ic(match[0])
                            ic(acc)
                            ic(evidence[0])
                            ic(u[0])
                            input("Press 'ENTER' to continue...")

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

            if scheduler is not None and phase == "train":
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

    time_elapsed = time.time() - since  # End time
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
