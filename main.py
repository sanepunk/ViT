import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import ViT
import sys
import numpy as np
from sklearn.metrics import accuracy_score
torch.set_num_threads(4)
model = ViT(
      image_size = 64,
			in_channels = 3,
			patch_size = 16,
			num_transformer_layers = 10,
			embedding_dim = 256,
			mlp_size = 3072,
			num_heads = 8,
			attention_dropout = 0.,
			mlp_dropout = 0.1,
			embedding_dropout = 0.1,
			num_classes= 100
) 
try:
    model.load_state_dict(torch.load('model_interrupted.pth'))
except:
    print("Model not available")

train_acc = []
test_acc = []
train_losses = []
test_losses = []
def handle_interrupt():
    print("\nTraining interrupted by user (Ctrl+C). Performing cleanup...")
    # Place any custom code here, like saving model, logging, etc.
    # For example, you could save the model state:
    torch.save(model.state_dict(), "model_interrupted.pth")
    np.savez('training_results.npz', train_acc, test_acc, train_losses, test_losses)
    print("Model state saved.")
    sys.exit(0)  # Exit the program gracefully




transform = transforms.Compose([
    transforms.Resize(64),  # Resize to 256x256 (standard ImageNet preprocessing)
    # transforms.CenterCrop(224),  # Crop the center to 224x224 (standard ImageNet size)
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10


try:
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        train_bar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training', ncols=100)

        for batch_idx, (inputs, labels) in enumerate(train_bar):

            optimizer.zero_grad()

            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predicted.cpu().numpy())
            # print("\nFirst 10 train labels:", list(all_train_labels[:10]))
            # print("\nFirst 10 train predictions:", list(all_train_preds[:10]))

            running_loss += loss.item()
            train_accuracy = accuracy_score(all_train_labels, all_train_preds) * 100
            train_bar.set_postfix(loss=running_loss / (batch_idx + 1), acc=train_accuracy)

            # Print training stats every 100 batches
            # if (batch_idx + 1) % 100 == 0:
            #     print(f'Batch [{batch_idx + 1}/{len(trainloader)}] - Loss: {running_loss / (batch_idx + 1):.4f} - Accuracy: {train_accuracy:.2f}%')

        # Save training loss and accuracy for each epoch
        epoch_train_loss = running_loss / len(trainloader)
        epoch_train_acc = accuracy_score(all_train_labels, all_train_preds) * 100
        train_losses.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        model.eval()  
        all_test_labels = []
        all_test_preds = []
        running_test_loss = 0.0

        with torch.no_grad():  
            for inputs, labels in testloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(predicted.cpu().numpy())

        epoch_test_loss = running_test_loss / len(testloader)
        epoch_test_acc = accuracy_score(all_test_labels, all_test_preds) * 100
        test_losses.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f} - Train Accuracy: {epoch_train_acc:.2f}%')
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Test Loss: {epoch_test_loss:.4f} - Test Accuracy: {epoch_test_acc:.2f}%')

    print("\nTraining completed!")
    print(f"Final Training Accuracy: {train_acc[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_acc[-1]:.2f}%")

except KeyboardInterrupt:
    handle_interrupt()