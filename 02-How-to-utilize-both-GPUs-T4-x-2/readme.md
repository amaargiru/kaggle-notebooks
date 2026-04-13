# Problem Description

Kaggle provides the ability to use a GPU accelerator directly from a notebook hosted on the site, and allows you to choose the GPU type — either P100 or T4 x 2. The T4 x 2 GPU theoretically enables faster neural network training, but its practical use comes with certain challenges.

In this article, we will proceed in the following order:  
• enable GPU P100 in Kaggle settings;  
&nbsp;&nbsp;&nbsp;&nbsp;• build a standard convolutional neural network to classify the MNIST dataset and measure its training speed;  
&nbsp;&nbsp;&nbsp;&nbsp;• check whether increasing the batch size affects training speed and how it impacts accuracy;  
• switch to GPU T4 x 2 in Kaggle settings;  
&nbsp;&nbsp;&nbsp;&nbsp;• train the neural network on the dual GPU without changing anything in the code, and see how it affects training speed;  
&nbsp;&nbsp;&nbsp;&nbsp;• discuss the DataParallel and DistributedDataParallel methods offered by PyTorch for working with multiple GPUs simultaneously;  
&nbsp;&nbsp;&nbsp;&nbsp;• rewrite the neural network training code using the Accelerate library (a wrapper over DistributedDataParallel), measure the training speedup, and discuss the imposed limitations.

<img src= "https://raw.githubusercontent.com/amaargiru/kaggle-notebooks/refs/heads/main/images/02-How-to-utilize-both-GPUs-T4-x-2/Two-head-GPU.jpg" alt ="Speed Up with T4 GPU" style='width: 600px;'>

# Preparatory Steps

Environment setup code:


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import os
import time
import datetime
import tqdm

import warnings
# Suppress warning about hardware and software compatibility
warnings.filterwarnings('ignore', category=UserWarning)
# Suppress warning about deprecated methods
warnings.filterwarnings('ignore', category=DeprecationWarning)
```


```python
# Check for the MNIST dataset
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

## GPU Configuration

**Important**: from this point on, you need to switch the GPU in Settings -> Accelerator to GPU P100. Further below there are more lines of text in bold where you will need to switch to GPU T4 x 2 or reset the current session. Running all the code using the "Run All" command will inevitably lead to errors.

Let's try:


```python
print(torch.cuda.is_available())

# GPU specifications
if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(torch.cuda.current_device()))
    print(torch.cuda.get_device_name(0))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'Using {device} device')
```


```python
!nvidia-smi
```

## Loading Datasets


```python
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

print(f"Train: {len(train_data)}, Test: {len(test_data)}")
```

# Single GPU

## CNN Code

A simple convolutional network for working with the MNIST dataset:


```python
class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

# Don't forget to use the GPU
model = CNN_Net().to(device)

print(f"\nModel structure: {model}\n\n")
```

## CNN Configuration

Choose the loss function:


```python
criterion = nn.CrossEntropyLoss()
```

Choose the optimizer:


```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

## Training

**Important**: right now the P100 GPU may not be working ([Error when trying to run a neural network on the P100 GPU in the Kaggle environment](https://www.kaggle.com/discussions/questions-and-answers/688523)), but hopefully this will be fixed soon.


```python
# Number of epochs
n_epochs = 16

# Empty list for the future plot
train_loss_curve = []

def model_train():
    model.train()

    start = time.time()

    # Create a progress bar
    pbar = tqdm.tqdm(range(1, n_epochs + 1))

    for epoch in pbar:
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss/len(train_loader.dataset)
        train_loss_curve.append(train_loss)
        pbar.set_description(f'Epoch {epoch}  \tTraining Loss: {train_loss:.4f}')

    end = time.time()
    print(f"\nNeural network training time: {str(datetime.timedelta(seconds=round(end-start)))}")

model_train()
```

## Loss Plot

Let's plot the loss curve for visual comparison with the plot obtained with an increased batch size.


```python
plt.plot(train_loss_curve, linewidth=2)
plt.grid(True)
plt.show()
```

Let's measure the model's accuracy on the test data:


```python
def model_eval():
    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

model_eval()
```

Accuracy: 98.95%

## Increasing Batch Size

Increasing the batch size is one of the intuitively straightforward methods for speeding up neural network training. Each time, a larger data batch will be sent to the GPU, which reduces the overhead of loading and unloading the GPU. As we noted at the beginning of the article, the trade-off is lower accuracy.


```python
# Increase batch size
train_loader = DataLoader(train_data, batch_size=512, shuffle=True)

model = CNN_Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model_train()

# Restore batch to the original size
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
```


```python
model_eval()
```


```python
# A bit of trickery so that the right half of the plot also visually counts from zero
x_positions = list(range(0, n_epochs)) + list(range(n_epochs, 2 * n_epochs))

plt.plot(x_positions, train_loss_curve, linewidth=2)

x_ticks = list(range(0, 2 * n_epochs, n_epochs // 4))
x_labels = [str(i % n_epochs) for i in x_ticks]

plt.xticks(x_ticks, x_labels)

# Highlight the second half of the plot
plt.axvspan(n_epochs, 2 * n_epochs, color='lightgrey', alpha=0.3)

plt.grid(True)
plt.show()
```

It is clearly visible that increasing the batch size had a devastating effect on training accuracy. Of course, depending on the specific task at hand, adjusting the batch size can and should be done, but it must be done wisely.

# Dual GPU

**Important**: from this point on, you need to switch the GPU in Settings -> Accelerator from GPU P100 to GPU T4 x 2.

So, we can see how our CNN performs with MNIST on a single GPU P100: 16 epochs in 2 minutes 39 seconds with a final accuracy of 98.95%.

Now let's try using the two T4 GPUs available on Kaggle in the T4 x 2 configuration. The idea is simple: if one GPU is good, two are potentially faster. But is this true in practice? Let's find out.

GPU P100 vs GPU T4

| Specification | Nvidia P100 | Nvidia T4 |
|---|---|---|
| Architecture | Pascal | Turing |
| CUDA cores | 3584 | 2560 |
| Memory | 16 GB HBM2 | 16 GB GDDR6 |
| Memory bandwidth | ~ 732 GB/s | ~ 320 GB/s |
| FP32 performance | ~ 9.3 TFLOPS | ~ 8.1 TFLOPS |
| TDP | 250 W | 70 W |

The T4 is a card primarily oriented toward inference and energy efficiency. From the data above, it is clear that using the Nvidia T4 would definitely reduce Kaggle's electricity bills. But can we properly parallelize computations and speed up training without losing accuracy?

Let's repeat the GPU test:


```python
print(torch.cuda.is_available())

# GPU specifications
if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(torch.cuda.current_device()))
    print(torch.cuda.get_device_name(0))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'Using {device} device')
```


```python
!nvidia-smi
```

## Naive training on Two GPUs

Let's first establish a baseline and run the same unmodified code on Nvidia T4 x 2:


```python
model = CNN_Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model_train()
```


```python
model_eval()
```

So, when using only one GPU from the dual T4 setup, we see a substantial loss in training speed. It appears that the Turing architecture of the T4 GPU does not compensate for the reduced memory bandwidth. The takeaway: if your code is not optimized for multi-GPU parallelization, do not enable the dual GPU option in Kaggle notebook settings — you will get slower training for no reason.

## DataParallel

This section was supposed to contain a brief discussion of [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) usage along with detailed code based on this method, but if you visit the documentation page, you will see:

```
Warning

It is recommended to use DistributedDataParallel, instead of this class, to do multi-GPU training, even if there is only a single node. 
```

Well, what can I say? DataParallel works within a single process, using Python threads to manage multiple GPUs. The problem is that Python has the Global Interpreter Lock (GIL), which prevents threads from truly running in parallel. Furthermore, all coordination falls on GPU 0: it copies the model to the second GPU before each forward pass, gathers results back, computes the loss, performs the backward pass, and distributes updated weights. As a result, GPU 0 does the work of two, while GPU 1 spends most of its time idling — a classic bottleneck.

This will be especially noticeable on our task because the model itself is very small and processes small images. Computations on the GPU take very little time, while the overhead — copying model parameters to the second GPU, splitting batches, transferring tensors back and forth over the PCIe bus, synchronization — takes significantly more time. It's like splitting the task of "moving a pen from one desk to another" between two people: the coordination time will exceed the time of the actual work.

So, in general, I will not provide a code example for DataParallel to avoid mixing current code with historical excursions. I will limit myself to a remark that echoes the PyTorch documentation — use DistributedDataParallel.

## DistributedDataParallel

Unlike DataParallel, [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP) is a fully functional mechanism for utilizing the second GPU. We won't achieve a twofold speedup, but we will try to get a noticeable performance gain.

DistributedDataParallel has two powerful advantages over DataParallel — DDP launches a separate process for each GPU (instead of a single process with threads, so the GIL is no longer an issue), and it also uses [NCCL](https://developer.nvidia.com/nccl) — Nvidia's optimized library for inter-GPU communication.

I debugged the code based on DistributedDataParallel on Kaggle, but unfortunately, it is not only somewhat overcomplicated but also only works when launched as a separate process, which requires writing the code to a text file and running it from the command line (we will discuss this limitation below when reviewing the code using the Accelerate package).

```
AttributeError: Can't get attribute 'train' on <module '__main__' (<class '_frozen_importlib.BuiltinImporter'>)>  
W0407 09:59:08.879000 225 torch/multiprocessing/spawn.py:165] Terminating process 281 via signal SIGTERM

ProcessExitedException: process 0 terminated with exit code 1
```

The problem is that in Jupyter notebooks on Kaggle, mp.spawn does not work correctly due to peculiarities of \_\_main\_\_. To work around this issue, you need to use the approach of launching subprocesses via subprocess, but the main code has to be turned into a text string, which is, of course, extremely inconvenient for editing and debugging. I managed to launch and debug the code using "pure" DDP on Kaggle, but it doesn't look very readable, and it's all inside a text string. Fortunately, many people dislike all this boilerplate, so the community has long since created numerous wrappers around DDP that allow for transparent code modifications without sacrificing efficiency. We will use the Hugging Face [Accelerate](https://huggingface.co/docs/accelerate/index) library, which requires almost no code changes.

## Hugging Face Accelerate

Let's first try following the approach shown in [Multi-GPU and Accelerate](https://www.kaggle.com/code/muellerzr/multi-gpu-and-accelerate/notebook). The code for our case looks genuinely simple:


```python
from accelerate import Accelerator

accelerator = Accelerator()

model = CNN_Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Preparation before defining the training function
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

train_loss_curve = []

def accel_model_train():
    model.train()

    start = time.time()
    pbar = tqdm.tqdm(range(1, n_epochs + 1))

    for epoch in pbar:
        train_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)  # instead of loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_loss_curve.append(train_loss)
        pbar.set_description(f'Epoch {epoch}  \tTraining Loss: {train_loss:.4f}')

    end = time.time()
    print(f"\nElapsed time: {str(datetime.timedelta(seconds=round(end-start)))}")

accel_model_train()
```

The code remains practically identical to what we used at the very beginning; the only differences are `accelerator.prepare()` and `accelerator.backward()`. The code runs without errors, but unfortunately, **it only uses one GPU**, which is clearly visible in "Session Metrics".

<img src= "https://raw.githubusercontent.com/amaargiru/kaggle-notebooks/refs/heads/main/images/02-How-to-utilize-both-GPUs-T4-x-2/one-gpu-load.png" alt ="Single GPU load" style='width: 297px;'>

I tried running the code in the original "Multi-GPU and Accelerate" notebook, but unfortunately, I was unable to get the second GPU activated there either. What's going on?

This is perfectly normal behavior. Hugging Face Accelerate is a wrapper that simplifies writing code for distributed training, but it does not launch multiple processes by itself. It merely prepares the model, optimizer, and data loader for distributed mode, while the actual launch of multiple processes (one per GPU) must happen externally — via the accelerate launch command. When you simply run the code in a Kaggle notebook as a regular script, only one process starts, and Accelerate sees only one GPU, so the second one remains unused.

To utilize both GPUs on Kaggle, you need to use a multi-process launch mechanism. In a notebook, this can be done via the `notebook_launcher` function from the Accelerate library: you wrap your training code in a separate function and call `notebook_launcher(training_function, num_processes=2)`. It is important that the creation of Accelerator, the model, the optimizer, and the call to `accelerator.prepare()` all happen inside this function, not outside — because each process must initialize its own objects. Only then can Accelerate correctly distribute the workload between two GPUs. But unfortunately, this trick didn't work for me on Kaggle either; each time I received error messages like this:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```
```
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
```

It appears that on Kaggle, CUDA is initialized by the platform itself, implicitly, before our code runs. In that case, notebook_launcher with spawn won't help. If you managed to work around this issue, please let me know!

A reliable way to bypass Kaggle's limitations is to run the training as a separate script via subprocess:


```python
# Write the training script to a file
train_script = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator
import torch.optim as optim
import time
import datetime
import tqdm


class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


num_epochs = 16

accelerator = Accelerator()
accelerator.print("Device: " + str(accelerator.device))
accelerator.print("Num processes: " + str(accelerator.num_processes))
accelerator.print("Distributed type: " + str(accelerator.distributed_type))

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

model = CNN_Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

train_loss_curve = []


def accel_model_train():
    model.train()

    start = time.time()
    pbar = tqdm.tqdm(range(1, num_epochs + 1))

    for epoch in pbar:
        train_loss = 0.0

        for images, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_dataloader.dataset)
        train_loss_curve.append(train_loss)
        desc = "Epoch " + str(epoch) + "  \tTraining Loss: " + "{:.4f}".format(train_loss)
        pbar.set_description(desc)

    end = time.time()
    elapsed = str(datetime.timedelta(seconds=round(end - start)))
    accelerator.print("")
    accelerator.print("Elapsed time: " + elapsed)


accel_model_train()

accelerator.wait_for_everyone()
unwrapped = accelerator.unwrap_model(model)
accelerator.save(unwrapped.state_dict(), "mnist_cnn.pth")
accelerator.print("Training complete! Model saved to mnist_cnn.pth")
"""

with open("train_script.py", "w") as f:
    f.write(train_script)

print("train_script.py written successfully")
```


```python
# Launch distributed training on 2 GPUs
!accelerate launch --multi_gpu --num_processes 2 train_script.py
```

We finally managed to get the second GPU working!

<img src= "https://raw.githubusercontent.com/amaargiru/kaggle-notebooks/refs/heads/main/images/02-How-to-utilize-both-GPUs-T4-x-2/two-gpu-load.png" alt ="Double GPU load" style='width: 298px;'>

A training speedup of approximately 1.6x — a very good result!

As you can see, I managed to achieve a substantial performance gain while preserving accuracy and using fairly understandable code, but unfortunately, writing the code to a separate executable file could not be avoided.

The only way I can sweeten the pill is by using the %%writefile magic — it's a built-in Jupyter function that simply writes the entire cell contents to the specified file. It's a standard approach, no worse than write() — just shorter and more readable. Unfortunately, the trained neural network weights still have to be passed to the main code by writing a binary file to disk.

The code below is an exact copy of the text string code shown above — don't look for differences between them. Debugging is still difficult, but at least you can easily change variable names and look for typos in function names.


```python
%%writefile train_script.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator
import torch.optim as optim
import time
import datetime
import tqdm


class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


num_epochs = 16

accelerator = Accelerator()
accelerator.print("Device: " + str(accelerator.device))
accelerator.print("Num processes: " + str(accelerator.num_processes))
accelerator.print("Distributed type: " + str(accelerator.distributed_type))

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

model = CNN_Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

train_loss_curve = []


def accel_model_train():
    model.train()

    start = time.time()
    pbar = tqdm.tqdm(range(1, num_epochs + 1))

    for epoch in pbar:
        train_loss = 0.0

        for images, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_dataloader.dataset)
        train_loss_curve.append(train_loss)
        desc = "Epoch " + str(epoch) + "  \tTraining Loss: " + "{:.4f}".format(train_loss)
        pbar.set_description(desc)

    end = time.time()
    elapsed = str(datetime.timedelta(seconds=round(end - start)))
    accelerator.print("")
    accelerator.print("Elapsed time: " + elapsed)


accel_model_train()

accelerator.wait_for_everyone()
unwrapped = accelerator.unwrap_model(model)
accelerator.save(unwrapped.state_dict(), "mnist_cnn.pth")
accelerator.print("Training complete! Model saved to mnist_cnn.pth")
```


```python
!accelerate launch --multi_gpu --num_processes 2 train_script.py
```

# Conclusion

That's all for now. I hope this guide was helpful, and that it will now be a bit easier for you to speed up your neural network training using a dual GPU setup.

Please let me know in the comments on this notebook if you managed to work around the limitations associated with notebook_launcher.
