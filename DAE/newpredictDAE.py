import argparse
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import time
import psutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
import vai_q_onnx
from onnxruntime.quantization import shape_inference

# Define EnhancedDenoisingAutoencoder
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
        clean_data = pd.read_csv(clean_file, header=None).values
        noisy_data = pd.read_csv(noisy_file, header=None).values
        self.clean_signals = torch.tensor(clean_data, dtype=torch.float32).unsqueeze(1).unsqueeze(1)[:, :, :, :512]
        self.noisy_signals = torch.tensor(noisy_data, dtype=torch.float32).unsqueeze(1).unsqueeze(1)[:, :, :, :512]

    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        return self.noisy_signals[idx], self.clean_signals[idx]

def create_dataloader(clean_file, noisy_file, batch_size):
    dataset = RealSignalDataset(clean_file, noisy_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

class ResnetCalibrationDataReader(vai_q_onnx.CalibrationDataReader):
    def __init__(self, clean_file: str, noisy_file: str, batch_size: int = 32):
        super().__init__()
        self.iterator = iter(create_dataloader(clean_file, noisy_file, batch_size))

    def get_next(self) -> dict:
        try:
            inputs, _ = next(self.iterator)
            inputs = inputs[:1]  # Ensure the input size matches the model's expected input size (1, 1, 1, 512)
            return {"input": inputs.numpy()}
        except StopIteration:
            return None

def export_model_to_onnx(model, input_tensor, output_path):
    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        opset_version=13,
        input_names=['input'],
        output_names=['output']
    )
    print(f'Model exported to {output_path}')

def preprocess_onnx_model(input_model_path, preprocessed_model_path):
    shape_inference.quant_pre_process(
        input_model_path=input_model_path,
        output_model_path=preprocessed_model_path,
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False
    )
    print(f'Preprocessed model saved to {preprocessed_model_path}')

def quantize_model(preprocessed_model_path, output_model_path, clean_file, noisy_file):
    dr = ResnetCalibrationDataReader(clean_file, noisy_file, batch_size=32)
    vai_q_onnx.quantize_static(
        preprocessed_model_path,
        output_model_path,
        dr,
        quant_format=vai_q_onnx.QuantFormat.QDQ,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type=vai_q_onnx.QuantType.QUInt8,
        weight_type=vai_q_onnx.QuantType.QInt8,
        enable_dpu=True,
        extra_options={'ActivationSymmetric': True}
    )
    print('Calibrated and quantized model saved at:', output_model_path)

def evaluate_model(session, data_loader):
    total_mse = 0.0
    total_rmse = 0.0
    num_samples = 0
    all_outputs = []
    all_targets = []
    total_inference_time = 0.0
    total_memory_usage = 0.0

    input_name = session.get_inputs()[0].name

    with torch.no_grad():
        for inputs, targets in data_loader:
            for input_tensor, target_tensor in zip(inputs, targets):
                input_tensor = input_tensor.unsqueeze(0).numpy()  # Ensure the input size matches the model's expected input size (1, 1, 1, 512)
                
                # Measure inference time
                start_time = time.time()
                output = session.run(None, {input_name: input_tensor})
                inference_time = time.time() - start_time
                total_inference_time += inference_time

                # Measure memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                total_memory_usage += memory_info.rss

                output = np.array(output).squeeze()

                target_tensor = target_tensor.squeeze().numpy()
                mse = mean_squared_error(target_tensor, output)
                rmse = np.sqrt(mse)

                total_mse += mse
                total_rmse += rmse
                num_samples += 1

                all_outputs.append(output)
                all_targets.append(target_tensor)

    avg_inference_time = total_inference_time / num_samples
    avg_memory_usage = total_memory_usage / num_samples

    return total_mse / num_samples, total_rmse / num_samples, avg_inference_time, avg_memory_usage, np.vstack(all_outputs), np.vstack(all_targets)

def plot_examples(inputs, outputs, targets, num_examples=5):
    plt.figure(figsize=(15, num_examples * 3))
    for i in range(num_examples):
        plt.subplot(num_examples, 1, i + 1)
        plt.plot(inputs[i], label='Input')
        plt.plot(outputs[i], label='Output', linestyle='--')
        plt.plot(targets[i], label='Target', linestyle=':')
        plt.title(f'Example {i + 1}')
        plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Define paths
    input_model_path = "model.onnx"
    preprocessed_model_path = "preprocessed_model.onnx"
    output_model_path = "quantized_model.onnx"
    clean_file = './motion_free_signal.csv'
    noisy_file = './signal_with_motion_artifacts.csv'

    # Load the pre-trained model
    model = EnhancedDenoisingAutoencoder()
    model.load_state_dict(torch.load("model_state.pth", map_location=torch.device('cpu')))
    model.eval()
    input_tensor = torch.randn(1, 1, 1, 512)  # Ensure this matches your model's expected input shape
    export_model_to_onnx(model, input_tensor, input_model_path)

    # Preprocess the ONNX model
    preprocess_onnx_model(input_model_path, preprocessed_model_path)

    # Quantize the preprocessed model
    quantize_model(preprocessed_model_path, output_model_path, clean_file, noisy_file)

    # Load the quantized model for inference
    quantized_model = onnx.load(output_model_path)

    # providers = ['CPUExecutionProvider']
    # provider_options = [()]

    providers = ['VitisAIExecutionProvider']
    cache_dir = Path(__file__).parent.resolve()
    provider_options = [{
                  'config_file': 'vaip_config.json',
                'cacheDir': str(cache_dir),
                'cacheKey': 'denoise_cache'
           }]
    session = ort.InferenceSession(quantized_model.SerializeToString(), providers=providers, provider_options=provider_options)

    # Create the test data loader
    test_loader = create_dataloader(clean_file, noisy_file, batch_size=32)

    # Evaluate the model
    mse, rmse, avg_inference_time, avg_memory_usage, outputs, targets = evaluate_model(session, test_loader)
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'Average Inference Time: {avg_inference_time} seconds')
    print(f'Average Memory Usage: {avg_memory_usage / (1024 ** 2)} MB')  # Convert bytes to MB

    # Show examples
    plot_examples([test_loader.dataset[i][0].squeeze().numpy() for i in range(5)], outputs[:5], targets[:5])

if __name__ == '__main__':
    main()
