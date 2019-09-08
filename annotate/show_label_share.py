import pandas as pd


def main():
  df = pd.read_csv('labels.csv')
  print(df['label'].value_counts())


if __name__ == '__main__':
  main()
