# Bloom model

## Installation

<details>
  <summary>Installation (Linux)</summary>

### Activate the shark.venv Virtual Environment

To run the bloom model, activate the shark.venv virtual environment using the following command:

```shell
source shark.venv/bin/activate

# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
```

### Install dependencies

```shell
pip install transformers==4.21.2

### Install Torch-MLIR for Running the Model

Use the following command to install and use the Bloom-ops branch of Torch-MLIR:
```
Use this branch of Torch-MLIR for running the model: https://github.com/vivekkhandelwal1/torch-mlir/tree/bloom-ops


### Run bloom model

```shell
python bloom_model.py
```

To run the bloom model with different configurations, you can specify the runtime device, model config, and text prompt using the following command-line arguments:

- `--device <device string>`: Specifies the runtime device.
- `--config <config string>`: Specifies the model config.
- `--prompt <prompt string>`: Specifies the text prompt.

### Run the Bloom Model with Complete 176B Params

To run the bloom model with the complete 176B params configuration, use the following command:.

To run the complete 176B params bloom model, run the following command:
```shell
python bloom_model.py --config "bloom"
```
