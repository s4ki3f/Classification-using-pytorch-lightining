# Classification-using-pytorch-lightining
<br>
<div>
  <a href="https://colab.research.google.com/drive/1-w8pJmMihuTkjNZ4INm_CFB7y8iWmHUC#scrollTo=2tvVvrTU-tkW"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>
<br>

Weights & Biases
<!--- @wandbcode{pytorch-lightning-image-classification-colab} -->

# Image Classification using PyTorch Lightning ⚡️

We will build an image classification pipeline using PyTorch Lightning. We will follow this [style guide](https://pytorch-lightning.readthedocs.io/en/stable/starter/style_guide.html) to increase the readability and reproducibility of our code. A cool explanation of this available [here](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY).

## Setting up PyTorch Lightning and W&B 

For this Notbook, we need PyTorch Lightning(ain't that obvious!) and Weights and Biases.
```bash
! pip install pytorch-lightning --quiet
# install weights and biases
!pip install wandb --quiet
```
need these imports.
```python
import pytorch_lightning as pl
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import CIFAR10

import wandb
```

login to wandb account for monitoring
```python
wandb.login()
```

## 🔧 DataModule - The Data Pipeline we Deserve

DataModules are a way of decoupling data-related hooks from the LightningModule so you can develop dataset agnostic models.

It organizes the data pipeline into one shareable and reusable class. A datamodule encapsulates the five steps involved in data processing in PyTorch:
- Download / tokenize / process. 
- Clean and (maybe) save to disk.
- Load inside Dataset.
- Apply transforms (rotate, tokenize, etc…).
- Wrap inside a DataLoader.

Learn more about datamodules [here](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html). Let's build a datamodule for the Cifar-10 dataset. 

# Dataset used here is Cifar-10
