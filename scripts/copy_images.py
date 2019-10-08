import os
from shutil import copyfile
import pandas as pd
from tqdm import tqdm
from config import IMAGE_DIR, TRAIN_IMAGE_DIR, OUTPUT_DIR
from utils import empty_dir


def main():
    empty_dir(TRAIN_IMAGE_DIR)
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'labels.csv'))
    for filename in tqdm(df['filename']):
        src = os.path.join(IMAGE_DIR, filename)
        dst = os.path.join(TRAIN_IMAGE_DIR, filename)
        copyfile(src, dst)


if __name__ == '__main__':
    main()
