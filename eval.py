# --- Libraries
import os

import numpy as np
import torch
import torch.nn.functional as F  # Noqa
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable

from helpers import rotate_img, get_device
from losses import get_evidence_alpha
# ---


def calc_prob(output, method, alpha=None):
    _, preds = torch.max(output, 1)

    if method == "evidence":
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    elif method == "softmax":
        prob = F.softmax(output, dim=1)

    output = output.flatten()
    prob = prob.flatten()
    preds = preds.flatten()

    return output, preds, prob


def eval_single_image(model, img_path, num_classes, uncertainty=False, device=None):
    img = Image.open(img_path).convert("L")
    if not device:
        device = get_device()
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    img_tensor = trans(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    img_variable = img_variable.to(device)

    if uncertainty:
        output = model(img_variable)
        _, alpha = get_evidence_alpha(output)
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        output, preds, prob = calc_prob(output, "evidence", alpha)

        print("Predict:", preds[0])
        print("Probs:", prob)
        print("Uncertainty:", uncertainty)

    else:
        output = model(img_variable)
        _, preds, prob = calc_prob(output, "softmax")

        print("Predict:", preds[0])
        print("Probs:", prob)

    labels = np.arange(10)
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})

    plt.title(f"Classified as: {preds[0]}, Uncertainty: {uncertainty.item()}")

    axs[0].set_title("One")
    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")

    axs[1].bar(labels, prob.cpu().detach().numpy(), width=0.5)
    axs[1].set_xlim([0, 9])
    axs[1].set_ylim([0, 1])
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Classification Probability")

    fig.tight_layout()

    plt.savefig(f"./results/{os.path.basename(img_path)}")


def rotating_image_classification(
    model, img, filename, uncertainty=False, threshold=0.5, device=None
):
    if not device:
        device = get_device()
    num_classes = 10
    Mdeg = 180
    Ndeg = Mdeg // 10 + 1
    ldeg = []
    lp = []
    lu = []
    classifications = []

    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)

        nimg = np.clip(a=nimg, a_min=0, a_max=1)

        rimgs[:, i * 28: (i + 1) * 28] = nimg
        trans = transforms.ToTensor()
        img_tensor = trans(nimg)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        img_variable = img_variable.to(device)

        if uncertainty:
            output = model(img_variable)
            evidence, alpha = get_evidence_alpha(output)
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)

            output, preds, prob = calc_prob(output, alpha)

            classifications.append(preds[0].item())
            lu.append(uncertainty.mean())

        else:

            output = model(img_variable)
            _, preds, prob = calc_prob(output, "softmax")
            classifications.append(preds[0].item())

        scores += prob.detach().cpu().numpy() >= threshold
        ldeg.append(deg)
        lp.append(prob.tolist())

    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    c = ["black", "blue", "red", "brown", "purple", "cyan"]
    marker = ["s", "^", "o"] * 2
    labels = labels.tolist()
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [4, 1, 12]})

    for i in range(len(labels)):
        axs[2].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    if uncertainty:
        labels += ["uncertainty"]

        if device == torch.device("cuda:0"):
            lu = [x.item() for x in lu]

        axs[2].plot(ldeg, lu, marker="<", c="red")

    print(classifications)

    axs[0].set_title('Rotated "1" Digit Classifications')
    axs[0].imshow(1 - rimgs, cmap="gray")
    axs[0].axis("off")
    plt.pause(0.001)

    empty_lst = [classifications]
    axs[1].table(cellText=empty_lst, bbox=[0, 1.2, 1, 1])
    axs[1].axis("off")

    axs[2].legend(labels)
    axs[2].set_xlim([0, Mdeg])
    axs[2].set_ylim([0, 1])
    axs[2].set_xlabel("Rotation Degree")
    axs[2].set_ylabel("Classification Probability")

    plt.savefig(filename)
