import os
import pandas as pd
from config import OUTPUT_DIR


def main():
    print(pd.read_csv(os.path.join(OUTPUT_DIR, 'labels.csv'))['label'].value_counts())


if __name__ == '__main__':
    main()
