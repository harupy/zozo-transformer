import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from config import IMAGE_DIR, OUTPUT_DIR


def calc_transform_vector(clothe_vectors, person_vectors):
  clothe_med = np.median(clothe_vectors, axis=0, keepdims=True)
  person_med = np.median(person_vectors, axis=0, keepdims=True)
  diff = person_med - clothe_med
  return normalize(np.maximum(diff, 0))


def transform(vector, transform_vector):
  diff = normalize(vector) - transform_vector
  return normalize(np.maximum(diff, 0))


def similar_items(query_vector, vectors, limit=30):
  cos_sims = cosine_similarity(query_vector, vectors).ravel()
  top_indices = cos_sims.argsort()[::-1][:limit]
  top_sims = cos_sims[top_indices]
  return top_indices, top_sims


def read_image(filename):
  img_path = os.path.join(IMAGE_DIR, filename)
  return plt.imread(img_path)


def main():
  df = pd.read_csv(os.path.join(OUTPUT_DIR, 'predictions.csv')).sort_values('prediction')
  df_clothe = df[df['prediction'] == 0]
  df_person = df[df['prediction'] == 1]

  # equalize the number of samples in each class
  num_samples = min(len(df_clothe), len(df_person))
  df = pd.concat([
    df_clothe.sample(num_samples),
    df_person.sample(num_samples)
  ]).reset_index(drop=True)

  assert len(set(df['prediction'].value_counts().tolist())) == 1, 'The number o'

  is_clothe = df['prediction'] == 0
  vectors = df.drop(['file_name', 'prediction'], axis=1).values
  clothe_vectors = vectors[is_clothe]
  person_vectors = vectors[~is_clothe]

  # calculate the transform vector
  transform_vector = calc_transform_vector(clothe_vectors, person_vectors)
  vectors = normalize(vectors)

  ncols = 4
  nrows = 5

  # if ncols=4 and nrows=3, the output image tile looks like:
  #
  # A BBBB CCCC
  # D EEEE FFFF
  # G HHHH IIII
  #
  # (each capital letter represents an image)

  query = []
  baseline = []
  proposed = []

  for _ in range(nrows):
    # random sample a person image
    query_item = df[df['prediction'] == 1].sample(1)
    query_idx = query_item.index[0]

    # default query feature
    query_vector = vectors[query_idx].reshape(1, -1)

    # transformed query feature
    query_vector_transformed = transform(query_vector, transform_vector)

    # find similar items
    sim_indices_normal, _ = similar_items(query_vector, vectors, ncols + 1)
    sim_indices_transformed, _ = similar_items(query_vector_transformed, vectors, ncols)
    sim_items_normal = df['file_name'].iloc[sim_indices_normal]
    sim_items_transformed = df['file_name'].iloc[sim_indices_transformed]

    #
    query.append(read_image(sim_items_normal.iloc[0]))
    baseline.append(np.hstack([read_image(x) for x in sim_items_normal[1:]]))
    proposed.append(np.hstack([read_image(x) for x in sim_items_transformed]))

  query = np.vstack(query)
  baseline = np.vstack(baseline)
  proposed = np.vstack(proposed)

  # make subplots and adjust the column width
  fig, axes = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, ncols, ncols]})

  images = [query, baseline, proposed]
  titles = ['Query', 'Baseline', 'Proposed']
  colors = ['red', 'blue', 'green']
  for ax, img, title, color in zip(axes, images, titles, colors):
    ax.imshow(img)
    h, w = img.shape[:-1]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=color)

    for spine in ax.spines.values():
      spine.set_edgecolor(color)
      spine.set_linewidth(1)

  plt.tight_layout()
  plt.subplots_adjust(wspace=0.1, hspace=0)
  plt.savefig(os.path.join(OUTPUT_DIR, 'result.png'), dpi=200)
  plt.show()


if __name__ == '__main__':
  main()
