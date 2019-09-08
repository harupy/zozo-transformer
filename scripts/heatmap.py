import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from config import OUTPUT_DIR


def main():
  df = pd.read_csv(os.path.join(OUTPUT_DIR, 'predictions.csv')).sort_values('prediction')
  value_counts = df['prediction'].value_counts().to_dict()
  vectors = df.drop(['file_name', 'prediction'], axis=1).values
  vectors = normalize(vectors)

  # text positions
  border_index = (df['prediction'] == 0).sum()
  clothe_y = border_index // 2
  person_y = border_index + (len(df) - border_index) // 2
  x = vectors.shape[-1] // 2

  fontdict = {
    'color': 'grey',
    'alpha': 0.3,
    'size': 30,
  }

  common_text_args = dict(
    fontdict=fontdict,
    horizontalalignment='center',
    verticalalignment='center'
  )

  ax = sns.heatmap(vectors, cmap='YlGnBu')
  ax.axhline(border_index)
  ax.text(x, clothe_y, 'Clothe ({})'.format(value_counts[0]), **common_text_args)
  ax.text(x, person_y, 'Person ({})'.format(value_counts[1]), **common_text_args)
  ax.set_xticks([])
  ax.set_yticks([])
  plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap.png'), dpi=300)


if __name__ == '__main__':
  main()
