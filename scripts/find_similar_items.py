import re
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils import l2_norm
from config import IMAGE_DIR, OUTPUT_DIR


def compute_transform_vector(clothe_vectors, person_vectors):
    clothe_med = np.median(clothe_vectors, axis=0, keepdims=True)
    person_med = np.median(person_vectors, axis=0, keepdims=True)
    diff = person_med - clothe_med
    return l2_norm(np.maximum(diff, 0))


def transform(vector, transform_vector):
    diff = l2_norm(vector) - transform_vector
    return l2_norm(np.maximum(diff, 0))


def similar_items(query_vector, vectors, limit=30):
    cos_sims = cosine_similarity(query_vector, vectors).ravel()
    top_indices = cos_sims.argsort()[::-1][:limit]
    return top_indices, cos_sims[top_indices]


def read_image(filename):
    img_path = os.path.join(IMAGE_DIR, filename)
    return plt.imread(img_path)


def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'features.csv')).sort_values('prediction')

    layers = defaultdict(list)
    for col in df.columns:
        m = re.match(r'(\w+)_\d+', col)
        if m:
            layers[m.group(1)].append(col)

    is_clothe = df['prediction'] == 0
    df_clothe = df[is_clothe]
    df_person = df[~is_clothe]

    # balance the number of samples in each class
    num_samples = min(len(df_clothe), len(df_person))
    df_clothe = df_clothe.iloc[:num_samples]
    df_person = df_person.iloc[:num_samples]
    df = pd.concat([df_clothe, df_person], axis=0).reset_index(drop=True)
    filenames = df['filename']

    # drop unnecessary columns
    cols_to_drop = ['filename', 'prediction']
    df_clothe.drop(cols_to_drop, axis=1, inplace=True)
    df_person.drop(cols_to_drop, axis=1, inplace=True)
    df.drop(cols_to_drop, axis=1, inplace=True)

    transform_vector = compute_transform_vector(df_clothe.values, df_person.values)

    # compute transform vectors for each layer
    # transform_vectors = []
    # for layer, cols in layers.items():
    #     clothe_vectors = df_clothe[cols].values
    #     person_vectors = df_person[cols].values
    #     transform_vectors.append(compute_transform_vector(clothe_vectors, person_vectors))
    # transform_vector = np.hstack(transform_vectors)

    ncols = 4
    nrows = 5

    # if ncols = 4 and nrows = 3, the output image tile looks like:
    #
    # A BBBB CCCC
    # D EEEE FFFF
    # G HHHH IIII
    #
    # (each capital letter represents an image)

    query = []
    baseline = []
    proposed = []

    vectors = l2_norm(df.values)

    samples = df_person.sample(nrows, random_state=32)
    for query_vector in samples.values.reshape(nrows, 1, -1):
        # transform query feature
        query_vector_transformed = transform(query_vector, transform_vector)

        # find similar items
        sim_indices, _ = similar_items(query_vector, vectors, ncols + 1)
        sim_indices_transformed, _ = similar_items(query_vector_transformed, vectors, ncols)
        sim_items = filenames.iloc[sim_indices]
        sim_items_transformed = filenames.iloc[sim_indices_transformed]

        # store images
        query.append(read_image(sim_items.iloc[0]))
        baseline.append(np.hstack([read_image(x) for x in sim_items[1:]]))
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
