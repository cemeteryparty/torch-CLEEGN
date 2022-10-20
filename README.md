# torch-CLEEGN

The PyTorch implementation of CLEEGN, a convolutional neural network for online automatic EEG reconstruction

## Installation

```sh
$ git clone https://github.com/cemeteryparty/torch-CLEEGN.git
$ cd torch-CLEEGN/
```

### Activate environment ###

```sh
$ conda create --name ENV_NAME python=3.7
$ conda activate ENV_NAME
```

### Install Library ###

```sh
$ nvidia-smi  # Get the CUDA Version
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

#### Check GPU Support

```py
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
dev_id = torch.cuda.current_device()
print(torch.cuda.get_device_name(dev_id))
```

### Training Usage

```sh
$ python main.py --train-anno configs/TRIAL/set_train.json \
	--train-anno configs/TRIAL/set_valid.json \
	--config-path configs/TRIAL/config.json
```
