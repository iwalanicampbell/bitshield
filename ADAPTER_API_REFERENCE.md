# OneFlip Adapter API Reference

## Module: `oneflip_adapter`

Complete API documentation for the OneFlip → BitShield adapter.

---

## Classes

### `OneFlipQuantizedAdapter`

Main adapter class for bridging OneFlip models with BitShield.

#### Constructor

```python
OneFlipQuantizedAdapter(
    oneflip_model_path: str,
    dataset: str = 'CIFAR10',
    model_arch: str = 'resnet'
)
```

**Parameters:**
- `oneflip_model_path` (str): Path to OneFlip .pth checkpoint file
- `dataset` (str, optional): Dataset name. Determines:
  - Input shape (e.g., CIFAR10 → (1,3,32,32))
  - Number of classes (e.g., CIFAR10 → 10)
  - Valid values: CIFAR10, CIFAR100, ImageNet, GTSRB, STL10
- `model_arch` (str, optional): Model architecture name
  - Valid values: resnet, preactres, vgg, or custom
  - Used for naming outputs

**Raises:**
- `FileNotFoundError`: If `oneflip_model_path` doesn't exist

**Example:**
```python
adapter = OneFlipQuantizedAdapter(
    "saved_model/resnet_CIFAR10/clean_model_1_int8_state.pth",
    dataset="CIFAR10",
    model_arch="resnet"
)
```

---

#### Method: `load_model()`

Load OneFlip checkpoint and extract state dictionary.

```python
def load_model(self) -> Dict[str, Any]
```

**Returns:**
- Dict: Model state dictionary, handles various checkpoint formats:
  - Direct state dict
  - Wrapped in `state_dict` key
  - Wrapped in `model` key
  - Wrapped in other common keys

**Raises:**
- `ValueError`: If checkpoint format is not recognized

**Example:**
```python
state_dict = adapter.load_model()
print(f"Loaded {len(state_dict)} parameters")
```

---

#### Method: `get_quantization_metadata()`

Extract quantization parameters from checkpoint.

```python
def get_quantization_metadata(self) -> Dict[str, Any]
```

**Returns:**
Dict with quantization metadata:
```python
{
    'model_arch': 'resnet',           # str
    'dataset': 'CIFAR10',              # str
    'quantized': True,                 # bool
    'quant_bits': 8,                   # int (optional, if present in checkpoint)
    'quant_scale': 0.1234,             # float (optional)
    'quant_zero_point': 128,           # int (optional)
}
```

**Example:**
```python
metadata = adapter.get_quantization_metadata()
print(f"Quantization bits: {metadata.get('quant_bits', 'N/A')}")
print(f"Is quantized: {metadata['quantized']}")
```

---

#### Method: `export_to_onnx()`

Export OneFlip model to ONNX format for TVM compilation.

```python
def export_to_onnx(
    self,
    output_dir: str,
    model_constructor_fn: Optional[Callable] = None
) -> str
```

**Parameters:**
- `output_dir` (str): Directory to save ONNX file
  - Directory is created if it doesn't exist
  - Output filename: `{model_arch}_{dataset}_quantized.onnx`

- `model_constructor_fn` (callable, optional): Function to construct model architecture
  - If None, uses torchvision ResNet18
  - Signature: `model = fn(num_classes: int)`
  - Example: `lambda n: torchvision.models.resnet18(num_classes=n)`

**Returns:**
- str: Full path to exported ONNX file

**Side Effects:**
- Creates `output_dir` if needed
- Saves ONNX file to disk
- Prints confirmation message

**Raises:**
- `ImportError`: If torchvision not available and no constructor provided
- `RuntimeError`: If model loading fails (tries strict=False automatically)

**Example:**
```python
from torchvision import models

onnx_path = adapter.export_to_onnx(
    output_dir="./onnx_exports",
    model_constructor_fn=lambda n: models.resnet18(num_classes=n)
)
print(f"Exported to: {onnx_path}")
```

---

#### Method: `create_bitshield_config()`

Create BitShield configuration file for the model.

```python
def create_bitshield_config(self, output_path: str) -> str
```

**Parameters:**
- `output_path` (str): Full path to save JSON config file
  - Example: `oneflip_configs/resnet_CIFAR10_quantized.json`
  - Parent directories are created if needed

**Returns:**
- str: Path to created config file

**Config File Format:**
```json
{
    "model_name": "QResnet_CIFAR10",
    "dataset": "CIFAR10",
    "architecture": "resnet",
    "quantized": true,
    "input_shape": [1, 3, 32, 32],
    "input_channels": 3,
    "input_height": 32,
    "input_width": 32,
    "num_classes": 10,
    "source": "oneflip",
    "original_checkpoint": "/path/to/checkpoint.pth",
    "model_arch": "resnet",
    "quant_bits": 8
}
```

**Side Effects:**
- Creates parent directories if needed
- Writes JSON to disk
- Prints confirmation message

**Example:**
```python
config_path = adapter.create_bitshield_config(
    "oneflip_configs/resnet_CIFAR10_quantized.json"
)
print(f"Config created: {config_path}")
```

---

#### Method: `get_inference_fn()`

Get a callable function for running inference on the model.

```python
def get_inference_fn(self) -> Callable[[torch.Tensor], torch.Tensor]
```

**Returns:**
- Callable: Function with signature:
  ```python
  outputs = fn(batch_tensor: torch.Tensor) -> torch.Tensor
  ```
  - Input: Batch of images, shape (batch, 3, H, W), any device
  - Output: Model predictions, shape (batch, num_classes)
  - Automatically handles device placement
  - Runs in eval mode with no gradients

**Example:**
```python
inference_fn = adapter.get_inference_fn()

# Use for inference
import torch
batch = torch.randn(4, 3, 32, 32)  # 4 CIFAR10 images
predictions = inference_fn(batch)  # Shape: (4, 10)
print(predictions.shape)
```

---

#### Helper Methods (Private)

```python
def _get_input_shape(self) -> Tuple[int, ...]
```
Returns expected input shape: `(1, channels, height, width)`

```python
def _get_num_classes(self) -> int
```
Returns number of output classes based on dataset

---

## Functions

### `integrate_oneflip_to_bitshield()`

Complete integration workflow from OneFlip to BitShield.

```python
def integrate_oneflip_to_bitshield(
    oneflip_model_path: str,
    bitshield_project_dir: str,
    dataset: str = 'CIFAR10',
    model_arch: str = 'resnet',
    model_constructor_fn: Optional[Callable] = None
) -> Dict[str, Any]
```

**Parameters:**
- `oneflip_model_path` (str): Path to OneFlip .pth checkpoint
- `bitshield_project_dir` (str): BitShield project root directory
- `dataset` (str, optional): Dataset name
- `model_arch` (str, optional): Model architecture name
- `model_constructor_fn` (callable, optional): Custom model constructor

**Returns:**
Dict with integration results:
```python
{
    'onnx_path': 'oneflip_onnx_exports/resnet_CIFAR10_quantized.onnx',
    'config_path': 'oneflip_configs/resnet_CIFAR10_quantized.json',
    'metadata': {
        'model_arch': 'resnet',
        'dataset': 'CIFAR10',
        'quantized': True,
        'quant_bits': 8,
        # ... other metadata
    },
    'num_classes': 10,
    'input_shape': [1, 3, 32, 32]
}
```

**Side Effects:**
- Creates `oneflip_onnx_exports/` directory
- Creates `oneflip_configs/` directory
- Exports ONNX file
- Creates config JSON
- Prints progress messages

**Example:**
```python
result = integrate_oneflip_to_bitshield(
    "saved_model/resnet_CIFAR10/model.pth",
    "/path/to/bitshield",
    dataset="CIFAR10",
    model_arch="resnet"
)

print(result['onnx_path'])
print(result['config_path'])
print(result['num_classes'])
```

---

## Command-Line Interface

### Adapter Script

```bash
python oneflip_adapter.py [options]
```

**Options:**

```
-model_path PATH
    Path to OneFlip model checkpoint (required)

-output_dir DIR
    Output directory for ONNX export
    Default: ./oneflip_onnx_exports

-bitshield_dir DIR
    BitShield project directory (required for -integrate)

-dataset NAME
    Dataset name (default: CIFAR10)

-arch NAME
    Model architecture (default: resnet)

-integrate
    Run full integration (requires -bitshield_dir)

-onnx_only
    Only export to ONNX (default if no -integrate)
```

**Examples:**

```bash
# Export to ONNX only
python oneflip_adapter.py \
  -model_path model.pth \
  -output_dir ./exports

# Full integration
python oneflip_adapter.py \
  -model_path model.pth \
  -bitshield_dir /path/to/bitshield \
  -integrate

# Custom dataset/architecture
python oneflip_adapter.py \
  -model_path model.pth \
  -bitshield_dir /path/to/bitshield \
  -dataset CIFAR100 \
  -arch vgg \
  -integrate
```

---

## Usage Patterns

### Pattern 1: Basic Adapter Usage

```python
from oneflip_adapter import OneFlipQuantizedAdapter

# Create adapter
adapter = OneFlipQuantizedAdapter("model.pth", dataset="CIFAR10")

# Get quantization info
metadata = adapter.get_quantization_metadata()
print(f"Quantized: {metadata['quantized']}")

# Export to ONNX
onnx_path = adapter.export_to_onnx("./exports")

# Create config
config_path = adapter.create_bitshield_config("./configs/model.json")

print(f"ONNX: {onnx_path}")
print(f"Config: {config_path}")
```

### Pattern 2: Full Integration

```python
from oneflip_adapter import integrate_oneflip_to_bitshield

result = integrate_oneflip_to_bitshield(
    "model.pth",
    "/path/to/bitshield",
    dataset="CIFAR10",
    model_arch="resnet"
)

# Use results for next pipeline steps
onnx_path = result['onnx_path']
config_path = result['config_path']
num_classes = result['num_classes']
```

### Pattern 3: Custom Model Architecture

```python
from oneflip_adapter import OneFlipQuantizedAdapter
from my_models import CustomModel

def my_constructor(num_classes):
    return CustomModel(num_classes=num_classes, pretrained=False)

adapter = OneFlipQuantizedAdapter("model.pth")
onnx_path = adapter.export_to_onnx(
    "exports",
    model_constructor_fn=my_constructor
)
```

### Pattern 4: Batch Processing

```python
from pathlib import Path
from oneflip_adapter import integrate_oneflip_to_bitshield

model_dir = Path("saved_model")
for model_file in model_dir.rglob("*_int8_state.pth"):
    result = integrate_oneflip_to_bitshield(
        str(model_file),
        "/path/to/bitshield"
    )
    print(f"Processed: {result['onnx_path']}")
```

---

## Data Types

### Quantization Metadata Dictionary

```python
{
    'model_arch': str,           # Model architecture name
    'dataset': str,              # Dataset name
    'quantized': bool,           # Whether model is quantized
    'quant_bits': Optional[int], # Number of quantization bits (4 or 8)
    'quant_scale': Optional[float], # Quantization scale factor
    'quant_zero_point': Optional[int], # Quantization zero point
    'quant_flip_bit': Optional[int], # Bit index flipped for backdoor
}
```

### Integration Result Dictionary

```python
{
    'onnx_path': str,         # Path to exported ONNX file
    'config_path': str,       # Path to BitShield config JSON
    'metadata': Dict,         # Quantization metadata (see above)
    'num_classes': int,       # Number of output classes
    'input_shape': List[int], # Input tensor shape [1, C, H, W]
}
```

---

## Error Handling

### Common Exceptions

| Exception | Cause | Solution |
|-----------|-------|----------|
| `FileNotFoundError` | Model path doesn't exist | Check file path |
| `ImportError` | PyTorch/torchvision not installed | Install dependencies |
| `RuntimeError` | Model loading fails | Check checkpoint format |
| `ValueError` | Unsupported checkpoint format | Verify checkpoint structure |

### Example: Error Handling

```python
from oneflip_adapter import OneFlipQuantizedAdapter

try:
    adapter = OneFlipQuantizedAdapter("model.pth")
    onnx_path = adapter.export_to_onnx("exports")
except FileNotFoundError:
    print("ERROR: Model file not found")
except ImportError:
    print("ERROR: Required packages not installed")
except RuntimeError as e:
    print(f"ERROR: Model loading failed: {e}")
```

---

## Performance Notes

- **Model Loading:** ~0.5-1 sec
- **ONNX Export:** ~2-5 sec
- **Config Creation:** <0.1 sec
- **Full Integration:** ~3-6 sec

---

## Compatibility

### Tested With

- Python: 3.8+
- PyTorch: 2.0+
- TorchVision: 0.15+
- TVM: 0.12+
- ONNX: 1.12+

### Supported OneFlip Models

- ✅ ResNet18
- ✅ PreActResNet18
- ✅ VGG16
- ✅ Any custom architecture (with custom constructor)

### Supported BitShield Targets

- ✅ AVX2 (Intel x86-64)
- ✅ LLVM (generic)
- ✅ TVM quantization pipelines

---

## See Also

- `ONEFLIP_QUICK_START.md` - Quick reference
- `oneflip_quantized_integration_guide.md` - Comprehensive guide
- `run_oneflip_pipeline.py` - Pipeline orchestration
- BitShield documentation
- OneFlip documentation

