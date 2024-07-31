import torch
import torch.onnx
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import vai_q_onnx
from onnxruntime.quantization import shape_inference
import onnx
import onnxruntime as ort
import torchvision.models as models
from pathlib import Path
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import argparse

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
    calibration_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return calibration_loader

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
    inference_times = []
    memory_usages = []

    input_name = session.get_inputs()[0].name

    with torch.no_grad():
        for inputs, targets in data_loader:
            for input_tensor, target_tensor in zip(inputs, targets):
                input_tensor = input_tensor.unsqueeze(0).numpy()  # Ensure the input size matches the model's expected input size (1, 1, 1, 512)
                
                start_time = time.time()
                mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # Memory usage in MB
                
                output = session.run(None, {input_name: input_tensor})
                
                mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # Memory usage in MB
                inference_time = time.time() - start_time
                
                output = np.array(output).squeeze()

                target_tensor = target_tensor.squeeze().numpy()
                mse = mean_squared_error(target_tensor, output)
                rmse = np.sqrt(mse)

                total_mse += mse
                total_rmse += rmse
                num_samples += 1

                all_outputs.append(output)
                all_targets.append(target_tensor)
                inference_times.append(inference_time)
                memory_usages.append(mem_after - mem_before)

    avg_inference_time = np.mean(inference_times)
    avg_memory_usage = np.mean(memory_usages)
    return total_mse / num_samples, total_rmse / num_samples, np.vstack(all_outputs), np.vstack(all_targets), avg_inference_time, avg_memory_usage

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

def main(use_cpu):
    # Define paths
    input_model_path = "model.onnx"
    preprocessed_model_path = "preprocessed_model.onnx"
    output_model_path = "quantized_model.onnx"
    clean_file = './motion_free_signal.csv'
    noisy_file = './signal_with_motion_artifacts.csv'

    # Define the model structure
    class ResNet50Regressor(models.ResNet):
        def __init__(self):
            super(ResNet50Regressor, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3])
            self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.fc = torch.nn.Linear(2048, 512)  # Adjust the output size to match your target

        def forward(self, x):
            x = super(ResNet50Regressor, self).forward(x)
            return x

    # Load and export the model
    model = ResNet50Regressor()
    model.load_state_dict(torch.load("./top_models/model_epoch_14_score_0.0077.pth", map_location=torch.device('cpu')))
    model.eval()
    input_tensor = torch.randn(1, 1, 1, 512)  # Ensure this matches your model's expected input shape
    export_model_to_onnx(model, input_tensor, input_model_path)

    # Preprocess the ONNX model
    preprocess_onnx_model(input_model_path, preprocessed_model_path)

    # Quantize the preprocessed model
    quantize_model(preprocessed_model_path, output_model_path, clean_file, noisy_file)

    # Load the quantized model for inference
    quantized_model = onnx.load(output_model_path)
    cache_dir = Path(__file__).parent.resolve()
    
    if use_cpu:
        providers = ['CPUExecutionProvider']
        provider_options = []
    else:
        providers = ['VitisAIExecutionProvider']
        provider_options = [{
                    'config_file': 'vaip_config.json',
                    'cacheDir': str(cache_dir),
                    'cacheKey': 'modelcachekey'
                }]
    
    session = ort.InferenceSession(quantized_model.SerializeToString(), providers=providers, provider_options=provider_options)

    # Create the test data loader
    test_loader = create_dataloader(clean_file, noisy_file, batch_size=32)

    # Evaluate the model
    mse, rmse, outputs, targets, avg_inference_time, avg_memory_usage = evaluate_model(session, test_loader)
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'Average Inference Time: {avg_inference_time} seconds')
    print(f'Average Memory Usage: {avg_memory_usage} MB')

    # Show examples
    plot_examples([test_loader.dataset[i][0].squeeze().numpy() for i in range(5)], outputs[:5], targets[:5])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with quantized model")
    parser.add_argument('--use_cpu', action='store_true', help="Use CPU for inference")
    args = parser.parse_args()
    main(args.use_cpu)
