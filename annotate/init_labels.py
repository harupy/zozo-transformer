import pandas as pd


def make_empty_csv(fp, columns):
  pd.DataFrame(columns=columns).to_csv(fp, index=False)


def main():
  make_empty_csv('labels.csv', ['file_name', 'label'])


if __name__ == '__main__':
  main()
