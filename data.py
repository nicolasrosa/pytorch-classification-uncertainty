# --- Libraries
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
# ---

# --- Global variables
NUM_CLASSES = 10
# ---

# --- Training and validation datasets
data_train = MNIST("./data/mnist",
                   download=True,
                   train=True,
                   transform=transforms.Compose([transforms.ToTensor()]))

data_val = MNIST("./data/mnist",
                 train=False,
                 download=True,
                 transform=transforms.Compose([transforms.ToTensor()]))
# ---

# --- Training and validation dataloaders
dataloader_train = DataLoader(data_train, batch_size=1000, shuffle=True, num_workers=8)
dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=8)

dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}
# ---

# --- Sample data, Digit "1"
digit_one, _ = data_val[5]  # digit_one: (torch.Tensor, [1, 28,28], float32), label: (int)
# ---
