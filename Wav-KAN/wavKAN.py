 import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from KAN import *

# Define the wavelet types
#wavelet_types = ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon', 'bump', etc.] #It can include #all wavelet types
wavelet_types = ['mexican_hat', 'morlet', 'dog', 'shannon']

# Load MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Trials and Epochs
trials = 5
epochs_per_trial = 50

# Loop over each wavelet type
for wavelet in wavelet_types:
    all_train_losses, all_train_accuracies = [], []
    all_val_losses, all_val_accuracies = [], []
    print(f'Wavelet is {wavelet}')

    for trial in range(trials):
        print(f'Trial is {trial}')
        # Define model, optimizer, scheduler for each trial
        model = KAN([28 * 28, 32, 10], wavelet_type=wavelet)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        criterion = nn.CrossEntropyLoss()

        trial_train_losses, trial_val_losses = [], []
        trial_train_accuracies, trial_val_accuracies = [], []

        for epoch in range(epochs_per_trial):
            # Training
            train_loss, train_correct, train_total = 0.0, 0, 0
            model.train()
            #for images, labels in tqdm(trainloader):
            for images, labels in trainloader:
                images = images.view(-1, 28 * 28).to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(trainloader)
            train_acc = 100 * train_correct / train_total
            trial_train_losses.append(train_loss)
            trial_train_accuracies.append(train_acc)

            # Validation
            val_loss, val_correct, val_total = 0.0, 0, 0
            model.eval()
            with torch.no_grad():
                for images, labels in valloader:
                    images = images.view(-1, 28 * 28).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(valloader)
            val_acc = 100 * val_correct / val_total
            trial_val_losses.append(val_loss)
            trial_val_accuracies.append(val_acc)

            # Update learning rate
            scheduler.step()

        all_train_losses.append(trial_train_losses)
        all_train_accuracies.append(trial_train_accuracies)
        all_val_losses.append(trial_val_losses)
        all_val_accuracies.append(trial_val_accuracies)

    # Average results across trials and write to Excel
    avg_train_losses = pd.DataFrame(all_train_losses).mean().tolist()
    avg_train_accuracies = pd.DataFrame(all_train_accuracies).mean().tolist()
    avg_val_losses = pd.DataFrame(all_val_losses).mean().tolist()
    avg_val_accuracies = pd.DataFrame(all_val_accuracies).mean().tolist()

    results_df = pd.DataFrame({
        'Epoch': range(1, epochs_per_trial + 1),
        'Train Loss': avg_train_losses,
        'Train Accuracy': avg_train_accuracies,
        'Validation Loss': avg_val_losses,
        'Validation Accuracy': avg_val_accuracies
    })

    # Save the results
    # Save the results to an Excel file named after the wavelet type
    file_name = f'{wavelet}_results.xlsx'
    results_df.to_excel(file_name, index=False)

    print(f"Results saved to {file_name}.")
