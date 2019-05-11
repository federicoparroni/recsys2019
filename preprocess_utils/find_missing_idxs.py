import data
import numpy as np

def load_libsvm_data(path, list_size=25):
  """Returns features and labels in numpy.array."""

  def _parse_line(line):
    """Parses a single line in LibSVM format."""
    tokens = line.split("#")[0].split()
    assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
    label = float(tokens[0])
    qid = tokens[1]
    kv_pairs = [kv.split(":") for kv in tokens[2:]]
    features = {k: float(v) for (k, v) in kv_pairs}
    return qid, features, label

  qid_to_index = {}

  with open(path, "rt") as f:
    for line in f:
      qid, features, label = _parse_line(line)
      if qid not in qid_to_index:
        # Create index and allocate space for a new query.
        qid_to_index[qid] = 1
        print(qid)
  return qid_to_index


correct_tgt_idxs = np.array(data.target_indices('local', 'no_cluster'))

dictionary=load_libsvm_data('dataset/preprocessed/tf_ranking/no_cluster/full/no_pop/vali.txt')
wrong_tgt_idxs = np.array(list(dictionary.keys()))
missing_idxs = np.setdiff1d(correct_tgt_idxs, wrong_tgt_idxs)
print(f'missing:{missing_idxs}\n len_missing:{len(missing_idxs)}')