import os
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from config import IMAGE_DIR

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

    return {
      'input': self.transform(image) if self.transform else image,
      'index': idx
    }


def seed_everything(seed=0):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


def compose_transform():
  return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])


def load_model(path):
  model = models.resnet50(pretrained=True)
  num_features = model.fc.in_features
  model.fc = nn.Linear(num_features, 2)
  model.load_state_dict(torch.load(path))
  return model.to(device)


def predict(model, data_loader):
  model.eval()
  preds = []

  for batch in tqdm(data_loader):
    inputs = batch['input'].to(device)

    with torch.no_grad():
      outputs = model(inputs)
      _, preds_batch = torch.max(outputs, 1)
      preds.extend(preds_batch.cpu().numpy())

  return np.array(preds)


def main():
  model = load_model('zozo.model')
  layer2 = []
  layer3 = []
  avgpool = []

  def hook_layer2(module, input, output):
    layer2.append(output.mean(dim=(-2, -1)))

  def hook_layer3(module, input, output):
    layer3.append(output.mean(dim=(-2, -1)))

  def hook_avgpool(module, input, output):
    avgpool.append(output.squeeze())

  model.layer2[-1].register_forward_hook(hook_layer2)
  model.layer3[-1].register_forward_hook(hook_layer3)
  model.avgpool.register_forward_hook(hook_avgpool)

  limit = 30000
  df = pd.DataFrame({'file_name': os.listdir(IMAGE_DIR)}).sample(limit)
  transform = compose_transform()
  dataset = CustomDataset(df, IMAGE_DIR, transform)
  data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
  preds = predict(model, data_loader)

  layer2 = torch.cat(layer2).cpu().numpy()
  layer3 = torch.cat(layer3).cpu().numpy()
  avgpool = torch.cat(avgpool).cpu().numpy()

  layer2 = normalize(layer2, norm='l2')
  layer3 = normalize(layer3, norm='l2')
  avgpool = normalize(avgpool, norm='l2')

  df['prediction'] = preds
  layer2_cols = ['l2_{}'.format(idx) for idx in range(layer2.shape[-1])]
  layer3_cols = ['l3_{}'.format(idx) for idx in range(layer3.shape[-1])]
  avg_cols = ['avg_{}'.format(idx) for idx in range(avgpool.shape[-1])]
  df_layer2 = pd.DataFrame(layer2, columns=layer2_cols)
  df_layer3 = pd.DataFrame(layer3, columns=layer3_cols)
  df_avg = pd.DataFrame(avgpool, columns=avg_cols)
  pd.concat([
    df.reset_index(drop=True),
    df_layer2.reset_index(drop=True),
    df_layer3.reset_index(drop=True),
    df_avg.reset_index(drop=True)
  ], axis=1).to_csv('predictions.csv', index=False)


if __name__ == '__main__':
  main()
