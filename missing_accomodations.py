import os
import sys
sys.path.append(os.getcwd())

import data
import pandas as pd
from tqdm import tqdm

def find():
  found_ids = []
  
  for ref in tqdm(data.train_df()['reference'].values):
    try:
      v = int(ref)
      found_ids.append(v)
    except ValueError:
      continue
  
  for ref in tqdm(data.test_df()['reference'].values):
    try:
      v = int(ref)
      found_ids.append(v)
    except ValueError:
      continue

  found_ids = set(found_ids)
  accomod_known = set(map(int, data.accomodations_df()['item_id'].values))
  missing = found_ids.difference(accomod_known)

  print(missing)
  print(len(missing))
  return missing

if __name__ == "__main__":
    find()