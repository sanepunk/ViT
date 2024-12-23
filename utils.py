import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the training data transformations
train_transform = transforms.Compose([
	transforms.RandomResizedCrop(224),  # Resize then crop to 224x224
	transforms.RandomHorizontalFlip(),  # Random horizontal flip
	transforms.ToTensor(),              # Convert image to PyTorch tensor
	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],        # Mean for each channel
		std=[0.229, 0.224, 0.225]          # Standard deviation for each channel
	)  # Normalize
])

# Define the testing data transformations
test_transform = transforms.Compose([
	transforms.Resize(256),             # Resize to 256x256
	transforms.CenterCrop(224),         # Center crop to 224x224
	transforms.ToTensor(),              # Convert image to PyTorch tensor
	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],        # Mean for each channel
		std=[0.229, 0.224, 0.225]          # Standard deviation for each channel
	)  # Normalize
])


# Load the training dataset
train_dataset = datasets.CIFAR100(
	root='./data',
	train=True,
	download=True,
	transform=train_transform
)

# Load the testing dataset
test_dataset = datasets.CIFAR100(
	root='./data',
	train=False,
	download=True,
	transform=test_transform
)


batch_size = 32

# Training data loader
train_loader = DataLoader(
	dataset=train_dataset,
	batch_size=batch_size,
	shuffle=True,
	num_workers=4
)

# Testing data loader
test_loader = DataLoader(
	dataset=test_dataset,
	batch_size=batch_size,
	shuffle=False,
	num_workers=4
)


def save_model(model: nn.Module, path: str):
	torch.save(model.state_dict(), path)
