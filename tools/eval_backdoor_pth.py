import os
import sys
import argparse
import torch

# Ensure BitShield repo root is on sys.path so "dataman" etc. import correctly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import dataman as dm  # BitShield's data loader

# Point this to your OneFlip repo so we can import CustomNetwork
sys.path.append('/mnt/e/GithubReps/OneFlip')  # adjust if different
from export_backdoor_to_onnx import CustomNetwork  # uses CIFAR10 + resnet backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth-path', required=True,
                        help='Path to the backdoored .pth file')
    parser.add_argument('--quantization', default='int8',
                        choices=['int4', 'int8', 'none'])
    args = parser.parse_args()

    dataset = 'CIFAR10'
    backbone = 'resnet'
    image_size = 32
    num_classes = 10
    batch_size = 100

    quantization = None if args.quantization == 'none' else args.quantization

    print(f'Loading backdoor model from: {args.pth_path}')
    model = CustomNetwork(backbone, dataset, num_classes, quantization=quantization)
    state = torch.load(args.pth_path, map_location='cpu')

    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    model.load_state_dict(state)
    model.eval()

    loader = dm.get_benign_loader(dataset, image_size, 'val', batch_size)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            # CustomNetwork returns (features, logits)
            _, logits = model(x)
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()

    acc = correct / total
    print(f'Backdoor model CIFAR10 accuracy (PyTorch, no TVM/ONNX): {acc:.4f} ({acc*100:.2f}%)')


if __name__ == '__main__':
    main()