# Описание проблемы

Kaggle предоставляет возможность пользоваться графическим ускорителем прямо при работе из ноутбука, расположенного на сайте, и позволяет выбрать тип GPU - либо P100, либо T4 x 2. GPU T4 x 2, теоретически, позволяет добиться ускорения обучения нейросетей, но его практическое использование сопряжено с некоторыми проблемами.

В этой статье мы будем действовать в следующей последовательности:  
• включим в настройках Kaggle GPU P100;  
&nbsp;&nbsp;&nbsp;&nbsp;• построим типовую свёрточную нейронную сеть для распознавания набора MNIST и замерим скорость её обучения;  
&nbsp;&nbsp;&nbsp;&nbsp;• проверим, влияет ли на скорость обучения увеличение размера батча и как это сказывается на точности;  
• переключимся в настройках Kaggle на GPU T4 x 2;  
&nbsp;&nbsp;&nbsp;&nbsp;• обучим нейронную сеть на сдвоенном GPU, ничего не меняя в коде, и посмотрим, как это отразится на скорости обучения;  
&nbsp;&nbsp;&nbsp;&nbsp;• обсудим методы DataParallel и DistributedDataParallel, предлагаемые PyTorch для одновременной работы с несколькими GPU;  
&nbsp;&nbsp;&nbsp;&nbsp;• перепишем код обучения нейронной сети с использованием библиотеки Accelerate (обёртка над DistributedDataParallel), измерим прирост скорости обучения и обсудим накладываемые ограничения.


# Подготовительные операции

Код подготовки рабочего окружения:


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
# Убираем предупреждение о необходимости соответствия аппаратного обеспечения и ПО
warnings.filterwarnings('ignore', category=UserWarning)
# Убираем предупреждение о необходимости актуализации методов
warnings.filterwarnings('ignore', category=DeprecationWarning)
```


```python
# Проверяем наличие датасета MNIST
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

## Конфигурация видеокарты

**Важно: начиная с этой точки, вам нужно переключить видеокарту в Settings -> Accelerator на GPU P100. Ниже есть еще строчки текста, выделенные жирным, там вам нужно будет переключиться уже на GPU T4 x 2 или сбросить текущую сессию. Запуск всего кода при помощи команды "Run All" обязательно приведёт к ошибкам.**

Попробуем:


```python
print(torch.cuda.is_available())

# Характеристики GPU
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

## Загрузка датасетов


```python
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

print(f"Train: {len(train_data)}, Test: {len(test_data)}")
```

# Одиночный GPU

## Код CNN

Простая свёрточная сеть для работы с датасетом MNIST.


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

# Не забываем использовать GPU
model = CNN_Net().to(device)

print(f"\nModel structure: {model}\n\n")

# Если хотите посмотреть, что у модели внутри, раскомментируйте код ниже
# for name, param in model.named_parameters():
#    print(f"Layer: {name} Size: {param.size()}  Values: {param[:2]} \n")
```

## Настройка CNN

Выбираем функцию потерь:


```python
criterion = nn.CrossEntropyLoss()
```

Выбираем функцию оптимизации:


```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

## Тренировка CNN


```python
# Количество эпох
n_epochs = 16

# Пустой список для будущего графика
train_loss_curve = []

def model_train():
    model.train()

    start = time.time()

    # Создаем прогресс-бар
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
    print(f"\nElapsed time: {str(datetime.timedelta(seconds=round(end-start)))}")

model_train()
```

Время обучения нейросети на GPU P100: 02:39

## График ошибки CNN

Отрисуем график ошибки для дальнейшего наглядного сравнения с графиком, полученным при увеличенном размере батча.


```python
plt.plot(train_loss_curve, linewidth=2)
plt.grid(True)
plt.show()
```

Измерим точность модели на тестовых данных:


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

## Увеличение размера батча

Увеличение размеров батча - один из интуитивно понятных методов ускорения обучения нейронной сети. Каждый раз на GPU будет передаваться пакет данных большего размера, что позволит уменьшить накладные расходы на загрузку и выгрузку GPU. Как мы и отмечали в начале статьи - расплатой будет меньшая точность.


```python
# Увеличиваем размеры батча
train_loader = DataLoader(train_data, batch_size=512, shuffle=True)

model = CNN_Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model_train()

# Возвращаем батч к исходному размеру
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
```

Время обучения на GPU P100 при большом размере батча: 01:48


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

Хорошо видно, что увеличение размеров батча отразилось на точности обучения самым плачевным образом. Разумеется, в зависимости от конкретной решаемой задачи регулировать размер батча можно и нужно, но делать это надо с умом.

# Сдвоенный GPU

**Важно: начиная с этой точки, вам нужно переключить видеокарту в Settings -> Accelerator с GPU P100 на GPU T4 x 2.**

Итак, мы видим, как наша CNN работает с MNIST на одиночном GPU P100: 16 эпох за 2 минуты 39 секунд с финальной точностью 98.95%.

Теперь попробуем использовать два GPU T4, доступных на Kaggle в конфигурации T4 x 2. Идея проста: если один GPU — хорошо, то два — потенциально быстрее. Но так ли это на практике? Давайте разберёмся.

GPU P100 vs GPU T4

| Характеристика | Nvidia P100 | Nvidia T4 |
|---|---|---|
| Архитектура | Pascal | Turing |
| CUDA-ядра | 3584 | 2560 |
| Память | 16 GB HBM2 | 16 GB GDDR6 |
| Пропускная способность памяти | ~ 732 GB/s | ~ 320 GB/s |
| FP32 производительность | ~ 9.3 TFLOPS | ~ 8.1 TFLOPS |
| TDP | 250W | 70W |

T4 — это карта, ориентированная в первую очередь на инференс и энергоэффективность. По приведённым данным видно, что используя Nvidia T4, мы явно уменьшим Kaggle счета на электроэнергию. Но удастся ли нам грамотно распараллелить вычисления и ускорить обучение, не теряя точности?

Повторяем тест GPU:


```python
print(torch.cuda.is_available())

# Характеристики GPU
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

## Тренировка на двух GPU

Давайте сначала зафиксируем точку отсчёта и запустим тот же самый неизмененный код на Nvidia T4 x 2:


```python
model = CNN_Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model_train()
```

Время обучения а GPU T4 x 2 без расспаралелливания PyTorch: 02:39.


```python
model_eval()
```

Accuracy: 99.16%.

Итак, при использовании только одного GPU из сдвоенного T4 мы видим существенную потерю в скорости обучения. Похоже, архитектура Turing GPU T4 не компенсирует снижение пропускной способности памяти. Вывод - если ваш код не оптимизирован для распараллеливания на несколько GPU, не включайте в настройках ноутбука Kaggle сдвоенную видеокарту, получите замедление обучения на ровном месте.

## DataParallel

Тут должен был быть небольшой рассказ про использование [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) и подробный код на основе этого метода, но если вы перейдёте на страницу документации, то увидите:

```
Warning

It is recommended to use DistributedDataParallel, instead of this class, to do multi-GPU training, even if there is only a single node. 
```

Ну что тут сказать? DataParallel работает внутри одного процесса, используя потоки Python для управления несколькими GPU. Проблема в том, что Python имеет глобальную блокировку интерпретатора (GIL), которая не позволяет потокам по-настоящему выполняться параллельно. Кроме того, вся координация ложится на GPU 0: именно он копирует модель на второй GPU перед каждым forward pass, собирает результаты обратно, считает loss, выполняет backward и рассылает обновлённые веса. Получается, что GPU 0 работает за двоих, а GPU 1 большую часть времени простаивает в ожидании — классическое узкое горлышко.

На нашей задаче это будет особенно заметно, потому что сама модель очень маленькая и обрабатывает маленькие изображения. Вычисления на GPU занимают очень мало времени, а вот накладные расходы — копирование параметров модели на второй GPU, разбиение батча, пересылка тензоров туда-обратно по шине PCIe, синхронизация — занимают существенно большее время. Это всё равно что разделить задачу «перенести ручку со стола на стол» между двумя людьми: время на координацию действий превысит время самой работы.

Так что, в общем, я не буду приводить пример кода для DataParallel, чтобы не перемежать актуальный код историческими экскурсами, ограничусь лишь ремаркой, повторяющей документацию PyTorch - используйте DistributedDataParallel.

## DistributedDataParallel

В отличие от DataParallel, [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP) - вполне рабочий механизм для задействования второго GPU. Двухкратное ускорение нам получить не удастся, но достичь заметного прироста производительности мы всё же попробуем.

В DistributedDataParallel есть два мощных преимущества перед DataParallel - DDP запускает отдельный процесс для каждого GPU (вместо одного процесса с потоками, так что GIL на уже не страшен), а также использует [NCCL](https://developer.nvidia.com/nccl) — оптимизированную библиотеку Nvidia для коммуникации между GPU.

Я отладил код на основе DistributedDataParallel на Kaggle, но, к сожалению, он не только несколько переусложнён, но и, к сожалению, тоже работоспособен только при условии запуска через отдельный процесс, что требует записи кода в текстовый файл с последующим запуском из командной строки (это ограничение мы рассмотрим ниже, при обсуждении кода с использованием пакета Accelerate) 

```
AttributeError: Can't get attribute 'train' on <module '__main__' (<class '_frozen_importlib.BuiltinImporter'>)>  
W0407 09:59:08.879000 225 torch/multiprocessing/spawn.py:165] Terminating process 281 via signal SIGTERM

ProcessExitedException: process 0 terminated with exit code 1
```

Проблема в том, что в Jupyter-ноутбуках на Kaggle mp.spawn не работает корректно из-за особенностей \_\_main\_\_. Для обхода этой проблемы нужно использовать подход с запуском подпроцессов через subprocess, но основной код придётся превратить в текстовую строку, что, конечно, крайне неудобно для редактирования и отладки. Мне удалось запустить и отладить код на "чистом" DDP на Kaggle, но выглядит он не очень понятно, да еще весь находится внутри текстовой строки. К счастью, весь это бойлерплейт не нравится многим, поэтому сообщество давно придумало множество обёрток над DDP, позволяющих прозрачно изменять код, не теряя эффективность. Мы воспользуемся библиотекой Hugging Face [Accelerate](https://huggingface.co/docs/accelerate/index), позволяющей практически не менять код.

## Accelerate

Попробуем сначала пойти по пути, показанному в [Multi-GPU and Accelerate](https://www.kaggle.com/code/muellerzr/multi-gpu-and-accelerate/notebook). Код для нашего случая выглядит действительно просто:


```python
from accelerate import Accelerator

accelerator = Accelerator()

model = CNN_Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Подготовка до определения функции обучения
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
            accelerator.backward(loss)  # вместо loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_loss_curve.append(train_loss)
        pbar.set_description(f'Epoch {epoch}  \tTraining Loss: {train_loss:.4f}')

    end = time.time()
    print(f"\nElapsed time: {str(datetime.timedelta(seconds=round(end-start)))}")

accel_model_train()
```

Код остался практически идентичным тому, который мы использовали в самом начале, отличия только в `accelerator.prepare()` и `accelerator.backward()`. Код не вызывает ошибки, но, к сожалению, **использует только один GPU**, что хорошо видно в "Session Metrics".

<img src= "https://raw.githubusercontent.com/amaargiru/kaggle-notebooks/refs/heads/main/images/02-How-to-utilize-both-GPUs-T4-x-2/one-gpu-load.png" alt ="Single GPU load" style='width: 297px;'>

Попробуйте запустить код в оригинальном ноутбуке "Multi-GPU and Accelerate", но мне, к сожалению, не удалось добиться активации второго GPU и там. Что происходит?

Это совершенно нормальное поведение. Hugging Face Accelerate — обёртка, которая упрощает написание кода для распределённого обучения, но сама по себе она не запускает несколько процессов. Она лишь подготавливает модель, оптимизатор и загрузчик данных к распределённому режиму, а фактический запуск нескольких процессов (по одному на каждый GPU) должен происходить извне — через команду `accelerate launch`. Когда вы просто запускаете код в ноутбуке Kaggle как обычный скрипт, стартует только один процесс, и Accelerate видит только один GPU, поэтому второй остаётся незадействованным.

Чтобы задействовать оба GPU на Kaggle, нужно использовать механизм запуска нескольких процессов. В ноутбуке это можно сделать через функцию `notebook_launcher` из библиотеки Accelerate: вы оборачиваете свой тренировочный код в отдельную функцию и вызываете `notebook_launcher(training_function, num_processes=2)`. При этом важно, чтобы создание `Accelerator`, модели, оптимизатора и вызов `accelerator.prepare()` происходили внутри этой функции, а не снаружи — потому что каждый процесс должен инициализировать свои собственные объекты. Только тогда Accelerate может корректно распределить нагрузку между двумя GPU. Но, к сожалению, такой финт на Kaggle у меня тоже не получился, каждый раз я получал сообщения об ошибках вроде этого:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```
```
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
```

Похоже, на Kaggle в ноутбуке CUDA инициализируется самой платформой, неявно, ещё до нашего кода. В таком случае notebook_launcher со spawn не поможет. Если вам удалось обойти эту проблему, пожалуйста, дайте мне знать!

Надёжный способ обойти ограничения Kaggle — запустить обучение как отдельный скрипт через subprocess:


```python
# Записываем скрипт тренировки в файл
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
# Запускаем распределённое обучение на 2 GPU
!accelerate launch --multi_gpu --num_processes 2 train_script.py
```

Наконец-то нам удалось добиться включения в работу второго GPU!

<img src= "https://raw.githubusercontent.com/amaargiru/kaggle-notebooks/refs/heads/main/images/02-How-to-utilize-both-GPUs-T4-x-2/two-gpu-load.png" alt ="Double GPU load" style='width: 298px;'>

Elapsed time: 01:40. Увеличение скорости обучения примерно в 1.7 раза, очень хороший результат!

Как видите, мне удалось добиться достаточно существенного прироста производительности с сохранением точности и использованием достаточно понятного кода, но, к сожалению, записи кода в отдельный исполняемый файл всё же избежать не получилось.

Единственное, чем я могу подсластить пилюлю - использованием магии %%writefile — это встроенная функция Jupyter, просто записывающей всё содержимое ячейки в указанный файл. Это штатный способ, ничем не хуже write() — просто короче и нагляднее. К сожалению, полученные коэффициенты нейронной сети всё так же придётся передавать в основной код через запись бинарного файла на диск.

Код ниже - это абсолютная копия кода текстовой строки, приведенной выше, не ищите в них отличий. Отладка всё также затруднена, но по крайней мере, можно легко менять названия переменных и искать опечатки в названиях функций.


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

# Заключение

На этом пока всё, надеюсь, это руководство было для вас полезным, и теперь вам будет немного проще ускорить обучение своей нейронной сети при помощи сдвоенного GPU.

Пожалуйста, дайте мне знать в комментариях к этому ноутбуку, если вам удалось обойти ограничения, связанные с notebook_launcher.
