import os
import random
from pathlib import Path
from shutil import copy2

VALID_SRC = Path("reid_data/valid")
QUERY_DST = Path("reid_data/query")
GALLERY_DST = Path("reid_data/gallery")

QUERY_DST.mkdir(parents=True, exist_ok=True)
GALLERY_DST.mkdir(parents=True, exist_ok=True)

for pid_dir in VALID_SRC.iterdir():
    if not pid_dir.is_dir():
        continue
    images = list(pid_dir.glob("*.jpg"))
    if len(images) < 2:
        continue  # need at least 1 for query + 1 for gallery

    random.shuffle(images)
    query_img = images[0]
    gallery_imgs = images[1:]

    pid = pid_dir.name
    os.makedirs(QUERY_DST / pid, exist_ok=True)
    os.makedirs(GALLERY_DST / pid, exist_ok=True)

    copy2(query_img, QUERY_DST / pid / query_img.name)
    for img in gallery_imgs:
        copy2(img, GALLERY_DST / pid / img.name)
