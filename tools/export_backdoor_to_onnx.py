import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import os
import argparse
import sys
import yaml

# ==============================================================================
# 1. CORE COMPONENTS FROM ONEFLIP/INJECT_BACKDOOR.PY
#    (These are needed to correctly instantiate and load your model)
# ==============================================================================

# Dummy import for PreActResNet18 if needed (modify path as necessary)
try:
    # Assuming the structure is relative to OneFlip/model_template
    sys.path.append(os.path.join(os.path.dirname(__file__), 'OneFlip/model_template'))
    from preactres import PreActResNet18 
except ImportError:
    # Define a simple placeholder if you only use ResNet from torchvision
    class PreActResNet18(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            print("WARNING: Using dummy PreActResNet18. Ensure proper source is imported.")
            self.linear = nn.Linear(512, num_classes)
        def forward(self, x): return self.linear(x)

class QuantizedLinear(nn.Module):
    """Quantized Linear layer definition (for int8 compatibility in saved weights)"""
    def __init__(self, in_features, out_features, num_bits=8):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        nn.init.constant_(self.bias, 0)
    
    def forward(self, x):
        # Only the floating point version is needed for ONNX export, 
        # as the quantization is baked into the model weights (.pth file)
        return nn.functional.linear(x, self.weight, self.bias)

class CustomNetwork(nn.Module):
    """Model definition replicating OneFlip/inject_backdoor.py logic."""
    def __init__(self, backbone, dataset, num_classes, quantization=None):
        super(CustomNetwork, self).__init__()
        self.quantization = quantization
        
        # NOTE: This model architecture loading logic must match your OneFlip setup
        if dataset == 'CIFAR10' and backbone == 'resnet':
            # Uses a torchvision ResNet18 modified to output 512 features
            self.model = torchvision.models.resnet18(weights=None, num_classes=512)
        elif dataset == 'GTSRB' and backbone == 'vgg':
            # Uses a torchvision VGG16 modified to output 512 features
            self.model = torchvision.models.vgg16(weights=None, num_classes=512)
        elif dataset == 'CIFAR100' and backbone == 'preactresnet':
            self.model = PreActResNet18(num_classes=512)
        else:
            raise ValueError(f"Unsupported combination: {backbone}/{dataset}")

        if quantization == 'int4':
            self.fc = QuantizedLinear(512, num_classes, num_bits=4)
        elif quantization == 'int8':
            self.fc = QuantizedLinear(512, num_classes, num_bits=8)
        else:
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.fc(x1)
        # Returns features and logits, as expected by the backdoor wrapper
        return x1, x2

class LogitWrapper(nn.Module):
    """Wrapper to expose only the final logits for ONNX export."""
    def __init__(self, base_model):
        super(LogitWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        _, logits = self.base_model(x)
        return logits

# ==============================================================================
# 2. STANDALONE EXPORT SCRIPT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export saved backdoored PT model to ONNX')
    parser.add_argument('--pth-path', type=str, required=True, 
                        help='Full path to the backdoored model .pth file.')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model name used by BitShield (must start with "Q", e.g., Qresnet50).')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset (e.g., CIFAR10).')
    parser.add_argument('--backbone', type=str, required=True,
                        help='Backbone name (e.g., resnet, vgg, preactresnet).')
    parser.add_argument('--quantization', type=str, default='int8', choices=['int4', 'int8', None],
                        help='Quantization type used for the saved model (matches CustomNetwork).')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size to embed in the ONNX filename/shape.')
    parser.add_argument('--image-size', type=int, default=32,
                        help='Input image size.')
    parser.add_argument('--output-root', type=str, 
                        default='bitshield/models', # Assumes this is where cfg.models_dir points
                        help='Root output directory (e.g., bitshield/models)')
    
    args = parser.parse_args()

    # Determine num_classes (needs to match your training setup)
    if args.dataset == 'CIFAR10':
        n_classes = 10
    elif args.dataset == 'CIFAR100':
        n_classes = 100
    elif args.dataset == 'GTSRB':
        n_classes = 43
    else:
        print("ERROR: Could not determine number of classes. Set manually or check CustomNetwork logic.")
        sys.exit(1)

    # 1. Instantiate the network architecture
    print(f"Instantiating {args.backbone} for {args.dataset} with {n_classes} classes...")
    model = CustomNetwork(
        args.backbone, 
        args.dataset, 
        n_classes, 
        quantization=args.quantization
    )
    model.eval()

    # 2. Load the backdoored weights
    try:
        model.load_state_dict(torch.load(args.pth_path, map_location=torch.device('cpu')))
        print(f"Successfully loaded weights from: {args.pth_path}")
    except Exception as e:
        print(f"ERROR loading state dict: {e}")
        sys.exit(1)

    # 3. Define output path structure (BitShield standard)
    onnx_dir = os.path.join(args.output_root, args.dataset, args.model_name)
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, f"{args.model_name}-{args.batch_size}.onnx")

    # 4. Prepare for ONNX export
    wrapped_model = LogitWrapper(model)
    # Input shape: (Batch Size, Channels=3, Height, Width)
    dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size)

    # 5. Export to ONNX
    print(f"\nExporting backdoored, quantized model to ONNX...")
    print(f"Output path: {onnx_path}")
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            input_names=['input0'],
            output_names=['output0'],
            opset_version=13, # Recommended opset for modern ONNX/TVM
            # Enable dynamic batch size, useful for BitShield testing
            dynamic_axes={'input0': {0: 'batch_size'}, 'output0': {0: 'batch_size'}} 
        )
        print("ONNX export completed successfully.")
        print(f"Your backdoored model is ready for the BitShield pipeline at: {onnx_path}")
    except Exception as e:
        print(f"FATAL ERROR during ONNX export: {e}")