# BitShield

## Overview

This is the research artifact for the paper [BitShield: Defending Against
Bit-Flip Attacks on DNN
Executables](https://www.ndss-symposium.org/ndss-paper/bitshield-defending-against-bit-flip-attacks-on-dnn-executables/)
in NDSS 2025.
It contains the code, data, and results for the paper and also enables future
research on the topic.

## Getting Started

### Setting Up

These external dependencies are needed for this project:

* [Pyenv](https://github.com/pyenv/pyenv#getting-pyenv)
* Docker

For the remaining dependencies, the first-time setup script will install them
automatically.

Run the first-time setup script:

```sh
cd debfd
./setup.sh
```

This script will

* Initialise all git submodules including the DL compilers
* Use Pyenv to install Python 3.8.12 and install the dependencies in the
  virtual environment `venv/`
* Build a Docker image `debfd-runner` so later tasks can be run in a stable
  and reproducible environment

### Noteworthy Notes

Remember to

```sh
source env.sh
```

before working on the project.

This project contains large files git is not good at tracking, e.g. datasets
and built binaries. They are tracked using [DVC](https://dvc.org/) which
should've been installed in the virtual environment. Every time new commits are
pulled, remember to run

```sh
dvc pull
```

as well to pull any new or modified data. Note that if you have new or modified
files in folders like `built/`, `datasets/`, or `results/sweep/`, this command
may **delete/overwrite** them!

To use the Docker image built earlier, run

```sh
docker/run-in-docker.sh <command>
```

## Usage

### Bridging from External Backdoor Models (ONNX Export)

If you already have a backdoored PyTorch model from another project, you can
export it to ONNX in the layout expected by BitShield using
tools/export_backdoor_to_onnx.py. This is the recommended entry point when
integrating BitShield with an existing backdoor pipeline.

1. Make sure your Python environment has PyTorch and torchvision installed
	 (these are already part of the standard BitShield setup).
2. Identify the following for your backdoored model:
	 - The `.pth` checkpoint file path.
	 - The dataset name (e.g. `CIFAR10`, `CIFAR100`, `GTSRB`).
	 - The backbone type used by your training code (one of `resnet`, `vgg`,
		 `preactresnet`).
	 - The BitShield model name you want to use (for consistency with the rest
		 of the pipeline we typically use names starting with `Q`, e.g.
		 `QresnetBD`).
	 - The quantization setting (`int4` or `int8`) that matches how the model
		 was trained/saved.
3. Run the export script, for example:

	 ```sh
	 python tools/export_backdoor_to_onnx.py \
		 --pth-path /path/to/backdoor_model.pth \
		 --model-name QresnetBD \
		 --dataset CIFAR10 \
		 --backbone resnet \
		 --quantization int8 \
		 --batch-size 32 \
		 --image-size 32 \
		 --output-root models
	 ```

	 This will produce an ONNX file at
	 models/<DATASET>/<MODEL_NAME>/<MODEL_NAME>-<BATCH>.onnx, e.g.
	 models/CIFAR10/QresnetBD/QresnetBD-32.onnx. This path is the standard
	 layout expected by cfg.models_dir and the BitShield build scripts.

Once the ONNX file is in place, you can treat the backdoored model like any
other BitShield model: add the corresponding configuration in cfg.py (if
needed), run dvc repro to build binaries and generate analysis, and then use
flipsweep.py and the tools/ scripts (such as catastrophic_stats.py and
defence_stats.py) to evaluate vulnerability and protection.

### Adding New Datasets and Editing Models

Usually dataset files live in `datasets/`, as configured in `cfg.py`. If your
new dataset is too large, you may consider symlinking it from somewhere else,
or modifying the config.

Other configurations like the models and compilation options are in `cfg.py` as
well.

Once your new configuration is ready, you may need to edit `dataman.py` to
expose it.

### Training Models

The definitions of models are in `support/models/`. To obtain the weights for a
model, you can use the training script `support/models/train.py`, for example:

```sh
for m in resnet50 densenet121 googlenet; do
	for x in CIFAR10 MNISTC FashionC; do
		support/models/train.py $m $x
	done
	for x in {0..11}; do
		support/models/train.py $m fake_$x
	done
done

support/models/train.py lenet1 MNIST --image-size 28

for x in {0..9}; do
	support/models/train.py lenet1 fakeM_$x --image-size 28
done
```

After training, you can generated the quantized models by running

```sh
docker/run-in-docker.sh support/models/quant.py
```

### Building Unprotected and Protected Models

After training, you can add the new model(s) to the build combinations in
`cfg.py`, and then run

```sh
dvc repro
```

which should build the binaries, import them into Ghidra, and generate the
sweep range information for later use.

Note that if you want to rebuild and/or redo the analysis done on a binary, you
need to **manually delete the existing files for them**. For the former you
just need to `rm` the files, but for the latter you may need to do it in Ghidra
GUI. Alternatively, you can discard the changes and restore them by running

```sh
git checkout -- dvc.lock
dvc checkout
```

### Performing the Vulnerable Bits Survey

Use `flipsweep.py` to brute force through all bits in compute functions and
look for vulnerable ones. For example, run

```sh
docker/run-in-docker.sh ./flipsweep.py -m resnet50 -d CIFAR10
```

to look for vulnerable bits in the ResNet50 binary trained on CIFAR10. You may
run this command on every sweep configuration you want to test.

This process can take a long time, but you can safely interrupt by pressing
`^C` twice. The resulting file is saved in `results/sweep/`. Note that if you
run `dvc checkout` later, the new file may get **deleted**.

### Reproducing Experiments

Once prior steps are done, you can reproduce the experiments in the paper by
running

```sh
tools/runattacksim.sh
```

which will run the attack simulation on all the models and datasets.
The results will be available in `results/`. Note that if you run `dvc
checkout` later, the new files may get **deleted**.
