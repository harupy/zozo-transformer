import os
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from utils import l2_norm
from config import IMAGE_DIR, OUTPUT_DIR

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


def predict(model, dataloader):
    model.eval()
    preds = []

    for batch in tqdm(dataloader):
        inputs = batch['input'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds_batch = torch.max(outputs, 1)
            preds.extend(preds_batch.cpu().numpy())

    return np.array(preds)


def main():
    clf = load_model(os.path.join(OUTPUT_DIR, 'zozo.model'))
    layer2 = []
    layer3 = []
    avgpool = []

    # define hoolks
    def hook_layer2(module, input, output):
        layer2.append(output.mean(dim=(-2, -1)))  # global average pooling by channel

    def hook_layer3(module, input, output):
        layer3.append(output.mean(dim=(-2, -1)))  # global average pooling by channel

    def hook_avgpool(module, input, output):
        avgpool.append(output.squeeze())  # remove the last two dimensions (height and width)

    # register hooks
    clf.layer2[-1].register_forward_hook(hook_layer2)
    clf.layer3[-1].register_forward_hook(hook_layer3)
    clf.avgpool.register_forward_hook(hook_avgpool)

    limit = 30000
    df = pd.DataFrame({'filename': os.listdir(IMAGE_DIR)}).sample(limit).reset_index(drop=True)
    transform = compose_transform()
    dataset = CustomDataset(df, IMAGE_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    preds = predict(clf, dataloader)

    # convert to numpy array
    layer2 = torch.cat(layer2, dim=0).cpu().numpy()
    layer3 = torch.cat(layer3, dim=0).cpu().numpy()
    avgpool = torch.cat(avgpool, dim=0).cpu().numpy()

    # normalize
    # layer2 = l2_norm(layer2)
    # layer3 = l2_norm(layer3)
    # avgpool = l2_norm(avgpool)

    l2_cols = ['l2_{}'.format(i) for i in range(layer2.shape[-1])]
    l3_cols = ['l3_{}'.format(i) for i in range(layer3.shape[-1])]
    avgp_cols = ['avgp_{}'.format(i) for i in range(avgpool.shape[-1])]

    # stack features and save it as a csv file
    pd.concat([
        df.assign(prediction=preds),
        pd.DataFrame(layer2, columns=l2_cols),
        pd.DataFrame(layer3, columns=l3_cols),
        pd.DataFrame(avgpool, columns=avgp_cols)
    ], axis=1).to_csv(os.path.join(OUTPUT_DIR, 'features.csv'), index=False)


if __name__ == '__main__':
    main()
