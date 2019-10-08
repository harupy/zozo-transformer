import sys
import os
import csv
import matplotlib.pyplot as plt
from config import IMAGE_DIR, OUTPUT_DIR
import pandas as pd


def make_empty_csv(fp, columns):
    pd.DataFrame(columns=columns).to_csv(fp, index=False)


def append_rows(fp, rows):
    with open(fp, 'a') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row if isinstance(row, list) else [row])


def fetch_unclassified(fp):
    classified = pd.read_csv(fp)['filename'].tolist()
    unclassified = sorted(list(set(os.listdir(IMAGE_DIR)) - set(classified)))
    return unclassified


def read_img(filename):
    return plt.imread(os.path.join(IMAGE_DIR, filename))


class Annotator:
    def __init__(self, outpath):
        self.outpath = outpath
        self.images = fetch_unclassified(self.outpath)
        self.num_images = len(self.images)
        self.classified = []  # store (image_name, label) pairs
        self.count = 1
        self.fig, self.ax = plt.subplots()
        self.axim = self.ax.imshow(read_img(self.images[-1]))

    def save(self):
        append_rows(self.outpath, self.classified)

    def start(self):
        self.ax.set_title(f'{self.count} / {self.num_images}')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def next(self):
        self.count += 1
        self.images.pop()

    def prev(self):
        if len(self.classified) > 0:
            self.count -= 1
            self.images.append(self.classified.pop()[0])

    def refresh(self):
        self.ax.set_title(f'{self.count} / {self.num_images}')
        self.axim.set_data(read_img(self.images[-1]))
        self.fig.canvas.draw()

    def on_key(self, event):
        mapping = {
            'p': 'person',
            'c': 'clothe',
            'k': 'unknown'
        }

        if event.key in mapping:
            self.classified.append([self.images[-1], mapping[event.key]])
            self.next()

        elif event.key == 'u':
            self.prev()

        elif event.key == 'q':
            print('Quitting...')
            self.save()
            plt.close('all')
            sys.exit()
        else:
            print('Invalid Key')

        self.refresh()


def main():
    outpath = os.path.join(OUTPUT_DIR, 'labels.csv')
    if not os.path.exists(outpath):
        make_empty_csv(outpath, ['filename', 'label'])

    annotator = Annotator(outpath)
    annotator.start()


if __name__ == '__main__':
    main()
