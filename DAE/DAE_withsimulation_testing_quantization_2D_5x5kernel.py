import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import pandas as pd
from pytorch_nndct.apis import torch_quantizer, Inspector
from sklearn.metrics import mean_squared_error

class EnhancedDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(EnhancedDenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1, 2))
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=(1, 2), mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def mean_squared_error_evaluation(model, data_loader):
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Reshape the tensors to 2D: [batch_size, signal_length]
            targets_reshaped = targets.squeeze(1).squeeze(1).cpu().numpy()
            outputs_reshaped = outputs.squeeze(1).squeeze(1).cpu().numpy()
            mse = mean_squared_error(targets_reshaped, outputs_reshaped)
            total_mse += mse
    return total_mse / len(data_loader)

def inspect_model(model, input_tensor):
    inspector = Inspector("AMD_AIE2_Nx4_Overlay")
    inspector.inspect(model, input_tensor)

def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedDenoisingAutoencoder().to(device)

    # Use the original dataset for training and calibration
    train_dataset = RealSignalDataset('./motion_free_signal.csv', './signal_with_motion_artifacts.csv')
    train_size = len(train_dataset) - 500
    calibration_size = 500
    train_subset, calibration_subset = random_split(train_dataset, [train_size, calibration_size])
    train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)

    # Extract calibration data
    calibration_loader = DataLoader(calibration_subset, batch_size=10)

    # Use the new datasets for evaluation
    test_dataset = RealSignalDataset('./motion_free_signal_test.csv', './signal_with_motion_artifacts_test.csv')
    test_loader = DataLoader(test_dataset, batch_size=10)

    # Inspect the model before quantization
    sample_input = torch.randn(10, 1, 1, 512).to(device)
    inspect_model(model, sample_input)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model.train()
    for epoch in range(500):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save trained model
    torch.save(model.state_dict(), 'model_state.pth')

    # Evaluate the model using new test data with MSE
    test_mse = mean_squared_error_evaluation(model, test_loader)
    print(f'Test MSE: {test_mse}')

    # Calibration using real data
    calibration_data = next(iter(calibration_loader))[0]  # Use the first batch for calibration
    quantizer = torch_quantizer('calib', model, (calibration_data.to(device),), device=device)
    quant_model = quantizer.quant_model
    quant_loss = evaluate(quant_model, calibration_loader, criterion)  # Calibration step
    print(f'Quantization Calibration Loss: {quant_loss}')

    # Export quantization results
    quantizer.export_quant_config()
    quantizer.export_xmodel(output_dir='./quantize_result', deploy_check=True)
    quantizer.export_onnx_model()
    print("Quantization completed and model exported.")

if __name__ == '__main__':
    main()
