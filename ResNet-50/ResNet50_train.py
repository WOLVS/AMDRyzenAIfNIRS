import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from torchvision.models import resnet50
import heapq
import os

class RealSignalDataset(Dataset):
    def __init__(self, clean_file, noisy_file):
        clean_data = pd.read_csv(clean_file, header=None)
        noisy_data = pd.read_csv(noisy_file, header=None)

        # Check the shape of the data
        print(f"Shape of clean_data: {clean_data.shape}")
        print(f"Shape of noisy_data: {noisy_data.shape}")

        # # Ensure the data is of shape (2808, 512)
        # if clean_data.shape != (2808, 512) or noisy_data.shape != (2808, 512):
        #     raise ValueError("Data shape is incorrect. Expected (2808, 512)")

        self.clean_signals = torch.tensor(clean_data.values, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        self.noisy_signals = torch.tensor(noisy_data.values, dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        print(f"Shape of clean_signals: {self.clean_signals.shape}")
        print(f"Shape of noisy_signals: {self.noisy_signals.shape}")

    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        return self.noisy_signals[idx], self.clean_signals[idx]

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_rmse = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs).view(-1, 1, 1, 512)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            targets_reshaped = targets.squeeze(1).squeeze(1).cpu().numpy()
            outputs_reshaped = outputs.squeeze(1).squeeze(1).cpu().numpy()
            mse = mean_squared_error(targets_reshaped, outputs_reshaped)
            rmse = np.sqrt(mse)
            total_mse += mse
            total_rmse += rmse
    return total_loss / len(data_loader), total_mse / len(data_loader), total_rmse / len(data_loader)

def save_top_models(top_models, model, epoch, score):
    if len(top_models) < 5:
        heapq.heappush(top_models, (score, epoch, model.state_dict().copy()))
    else:
        heapq.heappushpop(top_models, (score, epoch, model.state_dict().copy()))

def save_models(top_models):
    os.makedirs('top_models', exist_ok=True)
    for score, epoch, state_dict in top_models:
        model_path = f'top_models/model_epoch_{epoch}_score_{score:.4f}.pth'
        torch.save(state_dict, model_path)

def train_and_export_model():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, 512)  # Adjust for regression output

    model = model.to(device)

    train_dataset = RealSignalDataset('./motion_free_signal.csv', './signal_with_motion_artifacts.csv')
    train_size = len(train_dataset) - 500
    calibration_size = 500
    train_subset, calibration_subset = random_split(train_dataset, [train_size, calibration_size])
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

    test_dataset = RealSignalDataset('./motion_free_signal_test.csv', './signal_with_motion_artifacts_test.csv')
    test_loader = DataLoader(test_dataset, batch_size=32)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    top_models = []  # Min-heap to keep top 5 models

    model.train()
    for epoch in range(50):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)  # (N, 1, 1, 512)
            targets = targets.to(device)  # (N, 1, 1, 512)
            optimizer.zero_grad()
            outputs = model(inputs).view(-1, 1, 1, 512)  # Adjust output shape to (N, 1, 1, 512)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # Evaluate the model using test data
        test_loss, test_mse, test_rmse = evaluate(model, test_loader, criterion)
        score = test_mse + test_rmse  # Combined score with 1:1 weight
        save_top_models(top_models, model, epoch, score)
        print(f'Epoch {epoch+1}, Test MSE: {test_mse}, Test RMSE: {test_rmse}, Score: {score}')

    save_models(top_models)
    print("Top models saved.")

    dummy_input = torch.randn(1, 1, 1, 512).to(device)  # Adjust dimensions as needed
    torch.onnx.export(model, dummy_input, "model.onnx", opset_version=13, input_names=['input'], output_names=['output'])

if __name__ == '__main__':
    train_and_export_model()
