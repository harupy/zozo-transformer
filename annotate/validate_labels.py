import sys
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import cv2


def save_as_csv(path, data):
  pd.DataFrame(data).to_csv(path, index=False)


def append_rows(path, rows):
  with open(path, 'a') as f:
    writer = csv.writer(f)
    for row in rows:
      writer.writerow(row if isinstance(row, list) else [row])


def fetch_classified():
  df = pd.read_csv('labels.csv')
  return df[df['label'] == 'person']['file_name'].tolist()


class Checker:
  def __init__(self):
    self.imgs = fetch_classified()
    self.num_imgs = len(self.imgs)
    self.current_img = self.imgs.pop()
    self.count = 1
    self.fig, self.ax = plt.subplots()
    self.ims = self.ax.imshow(self.read_img(self.current_img))

  def read_img(self, img):
    return cv2.imread(os.path.join('../images', img))[:, :, ::-1]  # BGR to RGB

  def update(self):
    self.count += 1
    img = self.imgs.pop()
    self.current_img = img
    self.ax.set_title(f'{self.count} / {self.num_imgs}')
    self.ims.set_data(self.read_img(self.current_img))
    self.fig.canvas.draw()

  def on_key(self, event):
    if event.key == 'q':
      print('Quit')
      plt.close('all')
      sys.exit()

    self.update()

  def put_back(self):
    self.count -= 1
    self.imgs.append(self.current_img)

  def undo(self):
    self.count -= 1
    self.put_back()
    self.imgs.append(self.classified.pop()[0])

  def start(self):
    self.ax.set_title(f'{self.count} / {self.num_imgs}')
    self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    plt.show()


def main():
  checker = Checker()
  checker.start()


if __name__ == '__main__':
  main()
