import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hardening_layer_v3 as hl
from analyze_remaining_classes import deduplicate_pigeonhole, process_image

files = [
    os.path.join(r"D:\JMuBEN_Combined\Cerscospora", f)
    for f in os.listdir(r"D:\JMuBEN_Combined\Cerscospora")
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
print("Total Cerscospora files in JMuBEN_Combined:", len(files))

with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(process_image, files))

valid = [r for r in res if r[3] is None]
corrupt = [r for r in res if r[3] is not None]
print("Corrupt files:", len(corrupt))

phashes = [r[2] for r in valid]
acc_idx, dup_matches = deduplicate_pigeonhole(phashes, 4)
print("Unique Cerscospora at D <= 4:", len(acc_idx))
