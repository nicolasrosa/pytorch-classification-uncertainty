# --- Helpful links
# https://github.com/dougbrion/pytorch-classification-uncertainty
# http://douglasbrion.com/project/pytorch-classification-uncertainty
# ---

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

NUM_CLASSES = 10
ANNEALING_STEP = 10
# ---


def get_parser():
    parser = argparse.ArgumentParser()

    # --- Mode group
    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument("--train", action="store_true", help="To train the network.")
    mode_group.add_argument("--test", action="store_true", help="To test the network.")
    mode_group.add_argument("--examples", action="store_true", help="To example MNIST data.")
    # ---

    parser.add_argument("--epochs", default=10, type=int, help="Desired number of epochs.")
    parser.add_argument("--dropout", action="store_true", help="Whether to use dropout or not.")
    parser.add_argument("--uncertainty", action="store_true", help="Use uncertainty or not.")

    # --- Uncertainty type group
    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument(
        "--mse", dest='mse', action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error."
    )
    uncertainty_type_group.add_argument(
        "--digamma", dest='digamma', action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy."
    )
    uncertainty_type_group.add_argument(
        "--log", dest='log', action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood."
    )
    # ---

    return parser


def run_examples():
    examples = enumerate(dataloaders["val"])
    batch_idx, (example_data, example_targets) = next(examples)
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig("./images/examples.jpg")


def get_unc_loss(args):
    if args.digamma:
        return edl_digamma_loss
    elif args.log:
        return edl_log_loss
    elif args.mse:
        return edl_mse_loss


def get_traditional_loss():
    return nn.CrossEntropyLoss()


def get_loss(args, use_uncertainty):
    if use_uncertainty:
        return get_unc_loss(args)
    else:
        return get_traditional_loss()


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)


def get_scheduler(optimizer):
    return optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def run_train(args):
    # --- Local variables
    num_epochs = args.epochs
    use_uncertainty = args.uncertainty
    # ---

    # --- Define model, loss, optimizer and scheduler
    model = LeNet(NUM_CLASSES, dropout=args.dropout)
    model = model.to(DEVICE)

    criterion = get_loss(args, use_uncertainty)
    optimizer = get_optimizer(model)
    exp_lr_scheduler = get_scheduler(optimizer)
    # ---

    # --- Train model
    model, metrics = train_model(
        model,
        dataloaders,
        NUM_CLASSES,
        ANNEALING_STEP,
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


def get_checkpoint(args, use_uncertainty):
    checkpoint, filename = None, None

    if use_uncertainty:
        save_options = [
            ("digamma", args.digamma, "model_uncertainty_digamma.pt", "_uncertainty_digamma.jpg"),
            ("log", args.log, "model_uncertainty_log.pt", "_uncertainty_log.jpg"),
            ("mse", args.mse, "model_uncertainty_mse.pt", "_uncertainty_mse.jpg")
        ]
    else:
        save_options = [("model", True, ".pt", ".jpg")]

    for option, arg_value, checkpoint_suffix, filename_suffix in save_options:
        if arg_value:
            checkpoint = torch.load(f"./results/{checkpoint_suffix}")
            filename = f"./results/rotate{filename_suffix}"

    return checkpoint, filename


def run_test(args):
    # --- Local variables
    use_uncertainty = args.uncertainty
    # ---

    # --- Define model and optimizer(?)
    model = LeNet(NUM_CLASSES)
    model = model.to(DEVICE)
    # ---

    # --- Get checkpoint and restore it
    checkpoint, filename = get_checkpoint(args, use_uncertainty)

    model.load_state_dict(checkpoint["model_state_dict"])
    # ---

    # --- Set model to evaluation mode and test
    model.eval()

    rotating_image_classification(model, digit_one, filename, uncertainty=use_uncertainty)

    test_single_image(model, "./data/one.jpg", uncertainty=use_uncertainty)
    test_single_image(model, "./data/yoda.jpg", uncertainty=use_uncertainty)
    # ---


def check_args(parser):
    args = parser.parse_args()

    # Check if --uncertainty is provided without any of the required options
    if args.uncertainty and not (args.mse or args.log or args.digamma):
        parser.error("--uncertainty requires --mse, --log, or --digamma.")

    return args


def main():
    # --- Get arguments and check them
    parser = get_parser()
    args = check_args(parser)
    # ---

    # --- Run either train, test or examples
    if args.train:
        run_train(args)

    elif args.test:
        run_test(args)

    elif args.examples:
        run_examples()
    # ---

    print("\nDone!")


if __name__ == "__main__":
    main()
