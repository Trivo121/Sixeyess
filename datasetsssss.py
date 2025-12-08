import os
import json
import re
from pathlib import Path
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from collections import defaultdict

#loading the dataset
'''ds = load_dataset("BothBosu/multi-agent-scam-conversation")
dataset_dir = Path("D:/Study/Projects/sixeyessss/datasets/raw")
ds = load_dataset("BothBosu/multi-agent-scam-conversation", cache_dir=str(dataset_dir))
print(ds)'''

#parsed windows
RAW_DIR = Path("D:/Study/Projects/sixeyessss/datasets/raw")
BASE_OUT = Path("data/parsed_windows")
TRAIN_SCAM = BASE_OUT / "train" / "scam"
TRAIN_LEGIT = BASE_OUT / "train" / "legit"
TEST_SCAM = BASE_OUT / "test" / "scam"
TEST_LEGIT = BASE_OUT / "test" / "legit"


WINDOW_SIZE = 3
STEP = 1
PATTERN = re.compile(r"(Suspect:|Innocent:|caller:|receiver:|Caller:|Receiver:)", re.IGNORECASE)
def ensure_dirs():
    BASE_OUT.mkdir(parents=True, exist_ok=True)
TRAIN_SCAM.mkdir(parents=True, exist_ok=True)
TRAIN_LEGIT.mkdir(parents=True, exist_ok=True)
TEST_SCAM.mkdir(parents=True, exist_ok=True)
TEST_LEGIT.mkdir(parents=True, exist_ok=True)


def normalize_role(token):
t = token.strip().rstrip(':').lower()
if 'suspect' in t:
return 'Suspect'
if 'innocent' in t:
return 'Innocent'
if 'caller' in t:
return 'Caller'
if 'receiver' in t:
return 'Receiver'
return token.strip().rstrip(':')


def parse_to_turns(text):
if text is None:
return []
parts = PATTERN.split(text)
turns = []
for i in range(1, len(parts), 2):
if i + 1 < len(parts):
role = normalize_role(parts[i])
msg = parts[i+1].strip()
if msg:
turns.append(f"{role}: {msg}")
return turns


def build_windows_from_turns(turns):
windows = []
for i in range(0, len(turns), STEP):
chunk = turns[i:i+WINDOW_SIZE]
if len(chunk) < 2:
continue
text_chunk = ' | '.join(chunk)
windows.append(text_chunk)
return windows


def get_label_and_type(row):
label = None
scam_type = ''
if isinstance(row, dict):
if 'labels' in row:
try:
label = int(row['labels'])
except Exception:
label = 1 if row['labels'] else 0
elif 'label' in row:
try:
label = int(row['label'])
except Exception:
label = 1 if row['label'] else 0
for k in ('scam_type', 'scam_types', 'type', 'category'):
if k in row and row[k] is not None:
scam_type = str(row[k])
break