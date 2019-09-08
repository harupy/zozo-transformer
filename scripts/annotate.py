import sys
import os
import csv
import matplotlib.pyplot as plt
from config import IMAGE_DIR, LABEL_DIR
import pandas as pd
import cv2


def save_as_csv(path, data):
  pd.DataFrame(data).to_csv(path, index=False)


def append_rows(path, rows):
  with open(path, 'a') as f:
    writer = csv.writer(f)
    for row in rows:
      writer.writerow(row if isinstance(row, list) else [row])


def fetch_unclassified():
  classified = pd.read_csv(os.path.join(LABEL_DIR, 'labels.csv'))['file_name'].tolist()
  return sorted(list(set(os.listdir(IMAGE_DIR)) - set(classified)))


class Annotator:
  def __init__(self):
    self.imgs = fetch_unclassified()
    self.num_imgs = len(self.imgs)
    self.classified = []
    self.current_img = self.imgs.pop()
    self.count = 1
    self.fig, self.ax = plt.subplots()
    self.ims = self.ax.imshow(self.read_img(self.current_img))

  def read_img(self, img):
    return cv2.imread(os.path.join(IMAGE_DIR, img))[:, :, ::-1]  # BGR to RGB

  def update(self):
    self.count += 1
    img = self.imgs.pop()
    self.current_img = img
    self.ax.set_title(f'{self.count} / {self.num_imgs}')
    self.ims.set_data(self.read_img(self.current_img))
    self.fig.canvas.draw()

  def on_key(self, event):
    mapper = {
      'p': 'person',
      'c': 'clothe',
      'k': 'unknown'
    }

    if event.key in mapper:
      self.classified.append([self.current_img, mapper[event.key]])

    elif event.key == 'u':
      self.undo()

    elif event.key == 'q':
      print('Quitting...')
      self.save()
      plt.close('all')
      sys.exit()
    else:
      print('Invalid Key')
      self.put_back()  # put back the current image

    self.update()

  def put_back(self):
    self.count -= 1
    self.imgs.append(self.current_img)

  def undo(self):
    self.count -= 1
    self.put_back()
    self.imgs.append(self.classified.pop()[0])

  def save(self):
    append_rows(os.path.join(LABEL_DIR, 'labels.csv'), self.classified)

  def start(self):
    self.ax.set_title(f'{self.count} / {self.num_imgs}')
    self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    plt.show()


def main():
  annotator = Annotator()
  annotator.start()


if __name__ == '__main__':
  main()
