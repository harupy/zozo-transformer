import time
import os
import copy
import random
import json
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from config import TRAIN_IMAGE_DIR, OUTPUT_DIR


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
  def __init__(self, df, root_dir, transform=None):
    self.df = df
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    filename = self.df.iloc[idx, 0]
    img_path = os.path.join(self.root_dir, filename)
    image = Image.open(img_path)
    label = self.df.iloc[idx, 1]

    return {
      'input': self.transform(image) if self.transform else image,
      'label': label,
      'index': idx
    }


def seed_everything(seed=0):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


def train_model(model, data_loaders, data_sizes, criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()
  best_model_weights = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_corrects = 0
      cm = np.zeros((2, 2), dtype=np.int32)  # confusion matrix

      for idx, batch in enumerate(data_loaders[phase]):
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          if phase == 'train':
            loss.backward()
            optimizer.step()
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        cm += confusion_matrix(preds.cpu(), labels.data.cpu())
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / data_sizes[phase]
      epoch_acc = running_corrects.double() / data_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
      print('Confusion Matrix:\n', cm)

      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_weights = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best validation accuracy: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_weights)
  return model


def get_transforms():
  train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  return train, val


def main():
  seed_everything(0)

  # load labels and remove 'unknown'
  df = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_labels.csv'))
  df = df[df['label'].isin(['person', 'clothe'])]

  # encode labels
  le = LabelEncoder()
  df['label'] = le.fit_transform(df['label'])
  classes = {k: v for k, v in enumerate(le.classes_)}

  with open(os.path.join(OUTPUT_DIR, 'classes.json'), 'w') as f:
    f.write(json.dumps(classes))

  df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=0)

  # build data loader
  transform_train, transform_val = get_transforms()
  datasets = {
    'train': CustomDataset(df_train, TRAIN_IMAGE_DIR, transform_train),
    'val': CustomDataset(df_val, TRAIN_IMAGE_DIR, transform_val),
  }

  loader_settings = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 4
  }
  phases = ['train', 'val']
  data_loaders = {p: DataLoader(datasets[p], **loader_settings) for p in phases}
  data_sizes = {p: len(datasets[p]) for p in phases}

  # load ResNet50 and replace the last fully-connected layer
  model = models.resnet50(pretrained=True)
  num_features = model.fc.in_features
  model.fc = nn.Linear(num_features, 2)
  model = model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters())
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

  best_model = train_model(
    model,
    data_loaders,
    data_sizes,
    criterion,
    optimizer,
    exp_lr_scheduler,
    num_epochs=25
  )

  torch.save(best_model.state_dict(), os.path.join(OUTPUT_DIR, 'test.model'))


if __name__ == '__main__':
  main()
