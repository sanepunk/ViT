import torch
import torch.nn as nn
import torch.optim as optim
from models.ViT import ViT
from utils import save_model, train_loader, test_loader


# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for inputs, targets in loader:
		inputs, targets = inputs.to(device), targets.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

	epoch_loss = running_loss / total
	epoch_acc = correct / total
	return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for inputs, targets in loader:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)

			running_loss += loss.item() * inputs.size(0)
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	epoch_loss = running_loss / total
	epoch_acc = correct / total
	return epoch_loss, epoch_acc


num_epochs = 10  # Set the number of epochs
best_acc = 0.0


# Initialize the Vision Transformer model
model = ViT(
	image_size=224,
	in_channels=3,
	patch_size=16,
	num_transformer_layers=12,
	embedding_dim=768,
	mlp_size=3072,
	num_heads=12,
	attention_dropout=0.0,
	mlp_dropout=0.1,
	embedding_dropout=0.1,
	num_classes=100
).to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

for epoch in range(num_epochs):
	train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
	test_loss, test_acc = evaluate(model, test_loader, criterion, device)

	print(f'Epoch [{epoch+1}/{num_epochs}]')
	print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
	print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

	# Save the model if test accuracy improves
	if test_acc > best_acc:
		best_acc = test_acc
		save_model(model, f'{test_acc}_acc_vit_model.pth')

