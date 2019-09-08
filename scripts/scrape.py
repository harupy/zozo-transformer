import os
import uuid
import re
import shutil
import time
import multiprocessing

import urllib.parse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import IMAGE_DIR

if not os.path.exists(IMAGE_DIR):
  os.mkdir(IMAGE_DIR)


def empty_dir(dir_path):
  """ Remove files and directories in a specified directory """
  for fname in os.listdir(dir_path):
    fpath = os.path.join(dir_path, fname)
    if os.path.isfile(fpath):
      os.unlink(fpath)
    elif os.path.isdir(fpath):
      shutil.rmtree(fpath)


def gen_uuid():
  """ Generate random uuid """
  return str(uuid.uuid4)


def download_image(src):
  fname = os.path.basename(src)
  fpath = os.path.join(IMAGE_DIR, fname)
  try:
    resp = requests.get(src, stream=True)
    if resp.status_code == 200:
      with open(fpath, 'wb') as f:
        resp.raw.decode_content = True
        shutil.copyfileobj(resp.raw, f)
  except Exception as e:
    print(e.__class__.__name__, e)


def url_encode(query):
  return urllib.parse.quote(query.encode('shift-jis'))


def make_soup(url):
  resp = requests.get(url)
  return BeautifulSoup(resp.text, 'lxml')


def parse_img_src(soup):
  """ Extract image sources """
  return [
    re.sub(r'\d+.jpg$', '500.jpg', img.get('data-src'))
    for img in soup.select('img.catalog-img')
  ]


def main():
  empty_dir(IMAGE_DIR)

  # p_gtype=2 (old clothes) has a good balance of fitted and flat images.
  url_base = 'https://zozo.jp/category/tops/?p_gtype=2&pno={page_number}'
  num_pages = 500
  interval = 30  # seconds

  num_cpus = multiprocessing.cpu_count()

  for page_number in range(1, num_pages + 1):
    # scrape image src
    print(f'{page_number} / {num_pages}')
    url = url_base.format(page_number=page_number)
    soup = make_soup(url)
    srcs = parse_img_src(soup)

    # download images
    with multiprocessing.Pool(num_cpus) as p:
      with tqdm(total=len(srcs)) as t:
        for _ in p.imap_unordered(download_image, srcs):
          t.update(1)

    print(f'Waiting for {interval} seconds...')
    time.sleep(interval)


if __name__ == '__main__':
  main()
