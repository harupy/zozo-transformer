import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import l2_norm
from config import OUTPUT_DIR


def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'features.csv')).sort_values('prediction')
    value_counts = df['prediction'].value_counts().to_dict()

    # compute positions
    border_index = (df['prediction'] == 0).sum()
    clothe_y = border_index // 2
    person_y = border_index + (len(df) - border_index) // 2
    x = df.shape[0] // 2

    text_props = dict(
        fontdict={'color': 'grey', 'alpha': 0.3, 'size': 30},
        horizontalalignment='center',
        verticalalignment='center'
    )

    for layer in ['l2', 'l3', 'avgp']:
        print(f'Processing: {layer}')
        use_cols = [col for col in df.columns if col.startswith(layer)]
        vectors = l2_norm(df[use_cols].values[:, :])

        fig, ax = plt.subplots()
        ax = sns.heatmap(vectors, cmap='YlGnBu', ax=ax)
        ax.axhline(border_index)
        ax.text(x, clothe_y, 'Clothe ({})'.format(value_counts[0]), **text_props)
        ax.text(x, person_y, 'Person ({})'.format(value_counts[1]), **text_props)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_{layer}.png'), dpi=1000)


if __name__ == '__main__':
    main()
