import os
from shutil import copyfile
import pandas as pd
from tqdm import tqdm


def main():
  df = pd.read_csv('labels.csv')
  for filename in tqdm(df['file_name']):
    src = os.path.join('../images', filename)
    dst = os.path.join('../train_images', filename)
    copyfile(src, dst)


if __name__ == '__main__':
  main()
