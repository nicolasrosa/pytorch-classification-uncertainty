# --- Libraries
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from data import dataloaders, digit_one
from helpers import get_device
from lenet import LeNet
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss
from test import rotating_image_classification, test_single_image
from train import train_model
# ---

# --- Global variables
DEVICE = get_device()
print("Using device: {}".format(DEVICE))
# ---


def get_parser():
    parser = argparse.ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument("--train", action="store_true", help="To train the network.")
    mode_group.add_argument("--test", action="store_true", help="To test the network.")
    mode_group.add_argument("--examples", action="store_true", help="To example MNIST data.")

    parser.add_argument("--epochs", default=10, type=int, help="Desired number of epochs.")
    parser.add_argument("--dropout", action="store_true", help="Whether to use dropout or not.")
    parser.add_argument("--uncertainty", action="store_true", help="Use uncertainty or not.")

    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument("--mse", dest='mse', action="store_true",
                                        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.")
    uncertainty_type_group.add_argument("--digamma", dest='digamma', action="store_true",
                                        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.")
    uncertainty_type_group.add_argument("--log", dest='log', action="store_true",
                                        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.")

    return parser


def run_examples():
    examples = enumerate(dataloaders["val"])
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig("./images/examples.jpg")


def get_loss(parser, args, use_uncertainty):
    if use_uncertainty:
        if args.digamma:
            return edl_digamma_loss
        elif args.log:
            return edl_log_loss
        elif args.mse:
            return edl_mse_loss
        else:
            parser.error("--uncertainty requires --mse, --log or --digamma.")
    else:
        return nn.CrossEntropyLoss()


def run_train(parser, args):
    # --- Local variables
    num_epochs = args.epochs
    use_uncertainty = args.uncertainty
    num_classes = 10
    # ---

    # --- Define model, loss, optimizer and scheduler
    model = LeNet(dropout=args.dropout)
    model = model.to(DEVICE)

    criterion = get_loss(parser, args, use_uncertainty)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # ---

    # --- Train model
    model, metrics = train_model(
        model,
        dataloaders,
        num_classes,
        criterion,
        optimizer,
        scheduler=exp_lr_scheduler,
        num_epochs=num_epochs,
        device=DEVICE,
        uncertainty=use_uncertainty,
    )
    # ---

    # --- Get state dictionary
    state = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # --- Save model
    if use_uncertainty:
        save_options = [
            ("digamma", args.digamma),
            ("log", args.log),
            ("mse", args.mse)
        ]
        prefix = "./results/model_uncertainty_"

        for option, arg_value in save_options:
            if arg_value:
                save_path = prefix + option + ".pt"
                torch.save(state, save_path)
                print("Saved:", save_path)

    else:
        torch.save(state, "./results/model.pt")
        print("Saved: ./results/model.pt")
    # ---


def run_test(args):
    use_uncertainty = args.uncertainty
    model = LeNet()
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    if use_uncertainty:
        if args.digamma:
            checkpoint = torch.load("./results/model_uncertainty_digamma.pt")
            filename = "./results/rotate_uncertainty_digamma.jpg"
        if args.log:
            checkpoint = torch.load("./results/model_uncertainty_log.pt")
            filename = "./results/rotate_uncertainty_log.jpg"
        if args.mse:
            checkpoint = torch.load("./results/model_uncertainty_mse.pt")
            filename = "./results/rotate_uncertainty_mse.jpg"

    else:
        checkpoint = torch.load("./results/model.pt")
        filename = "./results/rotate.jpg"

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.eval()

    rotating_image_classification(model, digit_one, filename, uncertainty=use_uncertainty)

    test_single_image(model, "./data/one.jpg", uncertainty=use_uncertainty)
    test_single_image(model, "./data/yoda.jpg", uncertainty=use_uncertainty)


def main():
    # --- Get arguments
    parser = get_parser()
    args = parser.parse_args()
    # ---

    # --- Run either train, test or examples
    if args.train:
        run_train(parser, args)

    elif args.test:
        run_test(args)

    elif args.examples:
        run_examples()
    # ---


if __name__ == "__main__":
    main()
