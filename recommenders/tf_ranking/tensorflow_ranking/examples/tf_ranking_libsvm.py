# Copyright 2019 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""TF Ranking sample code for LETOR datasets in LibSVM format.

WARNING: All data sets are loaded into memory in this sample code. It is
for small data sets whose sizes are < 10G.

A note on the LibSVM format:
--------------------------------------------------------------------------
Due to the sparse nature of features utilized in most academic datasets for
learning to rank such as LETOR datasets, data points are represented in the
LibSVM format. In this setting, every line encapsulates features and a (graded)
relevance judgment of a query-document pair. The following illustrates the
general structure:

<relevance int> qid:<query_id int> [<feature_id int>:<feature_value float>]

For example:

1 qid:10 32:0.14 48:0.97  51:0.45
0 qid:10 1:0.15  31:0.75  32:0.24  49:0.6
2 qid:10 1:0.71  2:0.36   31:0.58  51:0.12
0 qid:20 4:0.79  31:0.01  33:0.05  35:0.27
3 qid:20 1:0.42  28:0.79  35:0.30  42:0.76

In the above example, the dataset contains two queries. Query "10" has 3
documents, two of which relevant with grades 1 and 2. Similarly, query "20"
has 1 relevant document. Note that query-document pairs may have different
sets of zero-valued features and as such their feature vectors may only
partly overlap or not at all.
--------------------------------------------------------------------------

Sample command lines:

OUTPUT_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train.txt && \
VALI=tensorflow_ranking/examples/data/vali.txt && \
TEST=tensorflow_ranking/examples/data/test.txt && \
rm -rf $OUTPUT_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/tf_ranking_libsvm_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_libsvm_py_binary \
--train_path=$TRAIN \
--vali_path=$VALI \
--test_path=$TEST \
--output_dir=$OUTPUT_DIR \
--num_features=136

You can use TensorBoard to display the training results stored in $OUTPUT_DIR.
"""

from absl import flags

import utils.check_folder as cf
import utils.telegram_bot as HERA
from recommenders.tf_ranking import TensorflowRankig
import numpy as np
import six
import tensorflow as tf
import tensorflow_ranking as tfr
import datetime
from utils.best_checkpoint_copier import BestCheckpointCopier
from random import randint
import sys
import pandas as pd
from tqdm import tqdm

_features = None
_labels = None
_features_vali = None
_labels_vali = None

class IteratorInitializerHook(tf.train.SessionRunHook):
  """Hook to initialize data iterator after session is created."""

  def __init__(self):
    super(IteratorInitializerHook, self).__init__()
    self.iterator_initializer_fn = None

  def after_create_session(self, session, coord):
    """Initialize the iterator after the session has been created."""
    del coord
    self.iterator_initializer_fn(session)

def all_equals(list):
    _RESULT = True
    current_el = list[0]
    for i in range(1,len(list),1):
        if list[i] != current_el:
            _RESULT = False
            break
    return _RESULT


def example_feature_columns():
  """Returns the example feature columns."""
  feature_names = ["{}".format(i + 1) for i in range(FLAGS.num_features)]
  return {
      name: tf.feature_column.numeric_column(
          name, shape=(1,), default_value=0.0) for name in feature_names
  }


def load_data(path, list_size):
    global flags_dict
    tf.logging.info("Loading data from {}".format(path))

    df = pd.read_hdf(path, key='df')

    # retrieve the qid and
    qid_list = df.pop('qid')
    labels = df.pop('label')
    assert (len(qid_list) == len(labels)), 'ATTENTION LEN OF QID AND LABEL_LIST IS NOT EQUAL!'

    # rename the columns with increasing numbers starting from 1
    tf.logging.info('Renaming columns')
    columns_names = df.columns
    new_names = np.arange(df.shape[1]) + 1
    dict_columns_names = dict(zip(columns_names, new_names))
    df.rename(columns=dict_columns_names, inplace=True)
    features_list = df.to_dict('records')

    # The 0-based index assigned to a query.
    qid_to_index = {}
    # The number of docs seen so far for a query.
    qid_to_ndoc = {}
    # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
    # a shape of [num_queries, list_size]. We use list for each of them due to the
    # unknown number of quries.
    feature_map = {k: [] for k in example_feature_columns()}
    label_list = []
    total_docs = 0
    discarded_docs = 0

    for i in tqdm(range(len(qid_list))):
        qid = qid_list[i]
        label = labels[i]
        features = features_list[i]

        if qid not in qid_to_index:
            # Create index and allocate space for a new query.
            qid_to_index[qid] = len(qid_to_index)
            qid_to_ndoc[qid] = 0
            for k in feature_map:
                feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
            label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
        total_docs += 1
        batch_idx = qid_to_index[qid]
        doc_idx = qid_to_ndoc[qid]
        qid_to_ndoc[qid] += 1
        # Keep the first 'list_size' docs only.
        if doc_idx >= list_size:
            discarded_docs += 1
            continue
        for k, v in six.iteritems(features):
            k = str(k)
            assert k in feature_map, "Key {} not found in features.".format(k)
            feature_map[k][batch_idx][doc_idx, 0] = v
        label_list[batch_idx][doc_idx] = label

    tf.logging.info("Number of queries: {}".format(len(qid_to_index)))
    tf.logging.info("Number of documents in total: {}".format(total_docs))
    tf.logging.info("Number of documents discarded: {}".format(discarded_docs))

    # Convert everything to np.array.

    context_features_id = []
    example_features_id = []
    for k in feature_map:
        feature_map[k] = np.array(feature_map[k])
        f_values = [el[0] for el in feature_map[k][0]]
        if k in flags_dict['context_features_id']:
            # convert the shape of the feature to a context feature shape
            feature_map[k] = feature_map[k][:, 0, :]
            context_features_id.append(k)
        else:
            example_features_id.append(k)
    print(context_features_id)

    _mode = path.split('/')[-1].split('.')[0]
    flags_dict[f'{_mode}_context_features_id'] = context_features_id
    flags_dict[f'{_mode}_per_example_features_id'] = example_features_id

    return feature_map, np.array(label_list)


def load_libsvm_data(path, list_size):
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

  tf.logging.info("Loading data from {}".format(path))

  # The 0-based index assigned to a query.
  qid_to_index = {}
  # The number of docs seen so far for a query.
  qid_to_ndoc = {}
  # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
  # a shape of [num_queries, list_size]. We use list for each of them due to the
  # unknown number of quries.
  feature_map = {k: [] for k in example_feature_columns()}
  label_list = []
  total_docs = 0
  discarded_docs = 0
  with open(path, "rt") as f:
    for line in f:
      qid, features, label = _parse_line(line)
      if qid not in qid_to_index:
        # Create index and allocate space for a new query.
        qid_to_index[qid] = len(qid_to_index)
        qid_to_ndoc[qid] = 0
        for k in feature_map:
          feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
        label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
      total_docs += 1
      batch_idx = qid_to_index[qid]
      doc_idx = qid_to_ndoc[qid]
      qid_to_ndoc[qid] += 1
      # Keep the first 'list_size' docs only.
      if doc_idx >= list_size:
        discarded_docs += 1
        continue
      for k, v in six.iteritems(features):
        assert k in feature_map, "Key {} not found in features.".format(k)
        feature_map[k][batch_idx][doc_idx, 0] = v
      label_list[batch_idx][doc_idx] = label

  tf.logging.info("Number of queries: {}".format(len(qid_to_index)))
  tf.logging.info("Number of documents in total: {}".format(total_docs))
  tf.logging.info("Number of documents discarded: {}".format(discarded_docs))

  # Convert everything to np.array.

  context_features_id = []
  example_features_id = []
  for k in feature_map:
    feature_map[k] = np.array(feature_map[k])
    f_values = [el[0] for el in feature_map[k][0]]
    if k in FLAGS.context_features_id:
        # convert the shape of the feature to a context feature shape
        feature_map[k] = feature_map[k][:, 0, :]
        context_features_id.append(k)
    else:
        example_features_id.append(k)
  print(context_features_id)

  _mode = path.split('/')[-1].split('.')[0]
  flags.DEFINE_list(f'{_mode}_context_features_id', context_features_id, 'all the key number of the context features')
  flags.DEFINE_list(f'{_mode}_per_example_features_id', example_features_id, 'all the key number of the per example features')

  return feature_map, np.array(label_list)


def get_train_inputs(features, labels, batch_size):
  """Set up training input in batches."""
  iterator_initializer_hook = IteratorInitializerHook()

  def _train_input_fn():
    """Defines training input fn."""
    features_placeholder = {
        k: tf.placeholder(v.dtype, v.shape) for k, v in six.iteritems(features)
    }
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder,
                                                  labels_placeholder))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    #1000
    iterator = dataset.make_initializable_iterator()
    feed_dict = {labels_placeholder: labels}
    feed_dict.update(
        {features_placeholder[k]: features[k] for k in features_placeholder})
    iterator_initializer_hook.iterator_initializer_fn = (
        lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
    return iterator.get_next()
  return _train_input_fn, iterator_initializer_hook

def get_batches(features, labels, batch_size):
  """Set up training input in batches."""
  iterator_initializer_hook = IteratorInitializerHook()

  def _train_input_fn():
    """Defines training input fn."""
    features_placeholder = {
        k: tf.placeholder(v.dtype, v.shape) for k, v in six.iteritems(features)
    }
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder,
                                                  labels_placeholder))

    dataset = dataset.batch(batch_size)

    #1000
    iterator = dataset.make_initializable_iterator()
    feed_dict = {labels_placeholder: labels}
    feed_dict.update(
        {features_placeholder[k]: features[k] for k in features_placeholder})
    iterator_initializer_hook.iterator_initializer_fn = (
        lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
    return iterator.get_next()
  return _train_input_fn, iterator_initializer_hook

def batch_inputs(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.batch(batch_size)

def make_score_fn():
  """Returns a groupwise score fn to build `EstimatorSpec`."""

  def _score_fn(unused_context_features, group_features, mode, unused_params,
                unused_config):
    """Defines the network to score a group of documents."""

    with tf.name_scope("input_layer"):
      """
      names = sorted(example_feature_columns())
      names.remove('28')
      group_input = [
          tf.layers.flatten(group_features[name])
          for name in names
      ]
      """
      per_ex_features = tf.concat([tf.layers.flatten(group_features[name]) for name in FLAGS.train_per_example_features_id],1)
      context_features = tf.concat([tf.layers.flatten(unused_context_features[name]) for name in FLAGS.train_context_features_id],1)

      input_layer = tf.concat([per_ex_features, context_features], axis=1)
      tf.summary.scalar("input_sparsity", tf.nn.zero_fraction(input_layer))
      tf.summary.scalar("input_max", tf.reduce_max(input_layer))

      tf.summary.scalar("input_min", tf.reduce_min(input_layer))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = tf.layers.batch_normalization(input_layer, training=is_training)
    for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
      cur_layer = tf.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.layers.batch_normalization(cur_layer, training=is_training)
      cur_layer = tf.nn.relu(cur_layer)
      tf.summary.scalar("fully_connected_{}_sparsity".format(i),
                        tf.nn.zero_fraction(cur_layer))
    cur_layer = tf.layers.dropout(
    cur_layer, rate=FLAGS.dropout_rate, training=is_training)
    logits = tf.layers.dense(cur_layer, units=FLAGS.group_size)
    return logits

  return _score_fn


def get_eval_metric_fns():
  """Returns a dict from name to metric functions."""
  metric_fns = {}
  metric_fns.update({
      "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
          tfr.metrics.RankingMetricKey.MRR,
      ]
  })
  return metric_fns


def train_and_eval():
  """Train and Evaluate."""

  global _features, _labels, _features_vali, _labels_vali
  if (_features is None) or (_labels is None):
      print('caching data train')
      _features, _labeals = load_libsvm_data(FLAGS.train_path, FLAGS.list_size)
  if (_features_vali is None) or (_labels_vali is None):
      print('caching data test')
      _features_vali, _labels_vali = load_libsvm_data(FLAGS.vali_path,
                                                FLAGS.list_size)

  train_input_fn, train_hook = get_train_inputs(_features, _labels,
                                                FLAGS.train_batch_size)

  vali_input_fn, vali_hook = get_batches(_features_vali, _labels_vali,
                                                FLAGS.train_batch_size)

  def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=FLAGS.learning_rate,
        optimizer="Adagrad")
  #Adagrad

  best_copier = BestCheckpointCopier(
      name='best',  # directory within model directory to copy checkpoints to
      checkpoints_to_keep=1,  # number of checkpoints to keep
      score_metric='metric/mrr',  # metric to use to determine "best"
      compare_fn=lambda x, y: x.score < y.score,
      # comparison function used to determine "best" checkpoint (x is the current checkpoint; y is the previously copied checkpoint with the highest/worst score)
      sort_key_fn=lambda x: x.score,
      sort_reverse=True,
      dataset_name=FLAGS.dataset_name,
      save_path=f'{FLAGS.save_path}',
      #save_path=f'dataset/preprocessed/tf_ranking/{_CLUSTER}/full/{_DATASET_NAME}/predictions',
      x=_features,
      y=_labels,
      test_x=_features_vali,
      test_y=_labels_vali,
      mode=FLAGS.mode,
      loss=FLAGS.loss,
      min_mrr_start=FLAGS.min_mrr_start,
      params=FLAGS
      )

  if FLAGS.loss == 'list_mle_loss':
      lambda_weight = tfr.losses.create_p_list_mle_lambda_weight(list_size=25)
  elif FLAGS.loss == 'approx_ndcg_loss':
      lambda_weight = tfr.losses.create_ndcg_lambda_weight()
  else:
      lambda_weight = tfr.losses.create_reciprocal_rank_lambda_weight()


  ranking_head = tfr.head.create_ranking_head(
      loss_fn=tfr.losses.make_loss_fn(FLAGS.loss, lambda_weight=lambda_weight),
      eval_metric_fns=get_eval_metric_fns(),
      train_op_fn=_train_op_fn)

  #weights_feature_name=FLAGS.weights_feature_number
  #lambda_weight=tfr.losses.create_reciprocal_rank_lambda_weight(smooth_fraction=0.5)

  estimator = tf.estimator.Estimator(
      model_fn=tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          group_size=FLAGS.group_size,
          transform_fn=tfr.feature.make_identity_transform_fn(FLAGS.train_context_features_id),
          #tfr.feature.make_identity_transform_fn(['28'])
          ranking_head=ranking_head),
    config=tf.estimator.RunConfig(
      FLAGS.output_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps))


  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      hooks=[train_hook],
      max_steps=FLAGS.num_train_steps)
  vali_spec = tf.estimator.EvalSpec(
      input_fn=vali_input_fn,
      hooks=[vali_hook],
      steps=None,
      exporters=best_copier,
      start_delay_secs=0,
      throttle_secs=30)

  # Train and validate
  tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)

def train_and_test():
    features, labels = load_libsvm_data(FLAGS.train_path, FLAGS.list_size)
    train_input_fn, train_hook = get_train_inputs(features, labels,
                                                  FLAGS.train_batch_size)
    features_test, labels_test = load_libsvm_data(FLAGS.test_path,
                                                  FLAGS.list_size)

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=FLAGS.learning_rate,
            optimizer="Adagrad")

    if FLAGS.loss == 'list_mle_loss':
        lambda_weight = tfr.losses.create_p_list_mle_lambda_weight(list_size=25)
    elif FLAGS.loss == 'approx_ndcg_loss':
        lambda_weight = tfr.losses.create_ndcg_lambda_weight(topn=25)
    else:
        lambda_weight = tfr.losses.create_reciprocal_rank_lambda_weight(topn=25)
    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(FLAGS.loss, lambda_weight=lambda_weight),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn)
    # tfr.losses.create_p_list_mle_lambda_weight(25)
    # lambda_weight=tfr.losses.create_reciprocal_rank_lambda_weight()

    estimator = tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=FLAGS.group_size,
            transform_fn=tfr.feature.make_identity_transform_fn(FLAGS.train_context_features_id),
            ranking_head=ranking_head))

    estimator.train(train_input_fn, hooks=[train_hook], steps=FLAGS.num_train_steps)

    # predict also for the train to get the scores for the staking
    pred_train = np.array(list(estimator.predict(lambda: batch_inputs(features, labels, 128))))
    pred = np.array(list(estimator.predict(lambda: batch_inputs(features_test, labels_test, 128))))


    pred_name_train=f'train_predictions_{FLAGS.loss}_learning_rate_{FLAGS.learning_rate}_train_batch_size_{FLAGS.train_batch_size}_' \
        f'hidden_layers_dim_{FLAGS.hidden_layer_dims}_num_train_steps_{FLAGS.num_train_steps}_dropout_{FLAGS.dropout_rate}_{FLAGS.group_size}'
    pred_name=f'predictions_{FLAGS.loss}_learning_rate_{FLAGS.learning_rate}_train_batch_size_{FLAGS.train_batch_size}_' \
        f'hidden_layers_dim_{FLAGS.hidden_layer_dims}_num_train_steps_{FLAGS.num_train_steps}_dropout_{FLAGS.dropout_rate}_{FLAGS.group_size}'
    np.save(f'{FLAGS.save_path}/{pred_name}', pred)
    np.save(f'{FLAGS.save_path}/{pred_name_train}', pred_train)

    for name in [pred_name, pred_name_train]:
        HERA.send_message(f'EXPORTING A SUB... mode:{FLAGS.mode}, name:{name}')
        model = TensorflowRankig(mode=FLAGS.mode, cluster='no_cluster', dataset_name=FLAGS.dataset_name,
                                 pred_name=name)
        model.name = f'tf_ranking_{name}'
        model.run()
        HERA.send_message(f'EXPORTED... mode:{FLAGS.mode}, name:{name}')


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.mode == 'full':
    train_and_test()
  else:
    train_and_eval()


if __name__ == "__main__":

    print('type mode: small, local or full')
    _MODE = input()
    print('type cluster')
    _CLUSTER=input()
    print('type dataset_name')
    _DATASET_NAME = input()
    print('insert the min_MRR from which export the sub')
    min_mrr = input()

    _BASE_PATH = f'dataset/preprocessed/tf_ranking/{_CLUSTER}/{_MODE}/{_DATASET_NAME}'
    _TRAIN_PATH = f'{_BASE_PATH}/train.txt'
    _TEST_PATH = f'{_BASE_PATH}/test.txt'
    _VALI_PATH = f'{_BASE_PATH}/vali.txt'

    # load context features id
    flags.DEFINE_list("context_features_id", list(np.load(f'{_BASE_PATH}/context_features_id.npy')), "id of the context features of the dataset")

    flags.DEFINE_float("min_mrr_start", min_mrr, "min_mrr_from_which_save_model")
    flags.DEFINE_string("save_path", _BASE_PATH, "path used for save the predictions")
    flags.DEFINE_string("train_path", _TRAIN_PATH, "Input file path used for training.")
    flags.DEFINE_string("vali_path", _VALI_PATH, "Input file path used for validation.")
    flags.DEFINE_string("test_path", _TEST_PATH, "Input file path used for testing.")
    flags.DEFINE_string("mode", _MODE, "mode of the models.")
    flags.DEFINE_string("dataset_name", _DATASET_NAME, "name of the dataset")
    # retrieve the number of features
    with open(f'{_BASE_PATH}/features_num.txt') as f:
        num_features = int(f.readline())
        print(f'num_features is: {num_features}')
    flags.DEFINE_integer("num_features", num_features, "Number of features per document.")
    flags.DEFINE_integer("list_size", 25, "List size used for training.")
    flags.DEFINE_integer('save_checkpoints_steps', 1000,
                         "number of steps after which save the checkpoint")

    print('1) random validation\n'
          '2) validation with hand-set parameters')
    res = input()
    if res == '1':
        def retrieve_random_params(dict):
            params = []
            for k,v in dict.items():
                params.append(v[randint(0, len(v)-1)])
            return params

        params_range_dict={
        'train_batch_size_choice':[8, 16, 32, 64, 128],
        'learning_rate_choice ':[0.01, 0.03, 0.08, 0.1, 0.15],
        'dropout_rate_choice ':[0.1, 0.2, 0.3, 0.4, 0.5],
        'hidden_layer_dims_choice':[['64', '32', '32'],
                                    ['128', '64'],
                                    ['256', '128'],
                                    ['64', '64', '64'],
                                    ['128', '64', '32'],
                                    ['64', '64', '64', '32', '32', '32'],
                                    ['128', '128']],
        'group_size_choice ':[1, 2, 5, 25],
        'loss_choice ':['pairwise_hinge_loss', 'pairwise_logistic_loss', 'pairwise_soft_zero_one_loss',
                'softmax_loss', 'approx_ndcg_loss'],
        }
        print('insert number of run of validator')
        times = int(input())
        for i in range(times):
            train_batch_size, learning_rate, dropout_rate, hidden_layer_dims, group_size, loss = retrieve_random_params(params_range_dict)
            if train_batch_size>128:
                num_train_steps = 100000
            else:
                num_train_steps = 200000
            cf.check_folder(f'{_BASE_PATH}/output_dir_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}')
            _OUTPUT_DIR = f'{_BASE_PATH}/output_dir_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}'
            if i == 0:
                flags.DEFINE_string("output_dir", _OUTPUT_DIR, "Output directory for models.")
                flags.DEFINE_integer("train_batch_size", train_batch_size, "The batch size for training.")
                flags.DEFINE_integer("num_train_steps", num_train_steps, "Number of steps for training.")
                flags.DEFINE_float("learning_rate", learning_rate, "Learning rate for optimizer.")
                flags.DEFINE_float("dropout_rate", dropout_rate, "The dropout rate before output layer.")
                flags.DEFINE_list("hidden_layer_dims", hidden_layer_dims,
                                  "Sizes for hidden layers.")
                flags.DEFINE_integer("group_size", group_size, "Group size used in score function.")
                flags.DEFINE_string("loss", loss,
                                    "The RankingLossKey for loss function.")
                FLAGS = flags.FLAGS
                FLAGS(sys.argv)
            else:
                FLAGS.set_default("output_dir", _OUTPUT_DIR)
                FLAGS.set_default("train_batch_size", train_batch_size)
                FLAGS.set_default("num_train_steps", num_train_steps)
                FLAGS.set_default("learning_rate", learning_rate)
                FLAGS.set_default("dropout_rate", dropout_rate)
                FLAGS.set_default("hidden_layer_dims", hidden_layer_dims)
                FLAGS.set_default("group_size", group_size)
                FLAGS.set_default("loss", loss)

            tf.logging.set_verbosity(tf.logging.INFO)
            train_and_eval()

    else:
        cf.check_folder(f'{_BASE_PATH}/output_dir_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}')
        _OUTPUT_DIR = f'{_BASE_PATH}/output_dir_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}'
        flags.DEFINE_string("output_dir", _OUTPUT_DIR, "Output directory for models.")
        # let the user choose the params
        print('insert train batch size')
        train_batch_size = input()
        print('insert learning rate')
        learning_rate = input()
        print('insert dropout rate')
        dropout_rate = input()
        print('insert hidden layer dims as numbers separeted by spaces')
        hidden_layer_dims = input().split(' ')
        print('select loss:\n'
              '1) PAIRWISE_HINGE_LOSS\n'
              '2) PAIRWISE_LOGISTIC_LOSS\n'
              '3) PAIRWISE_SOFT_ZERO_ONE_LOSS\n'
              '4) SOFTMAX_LOSS\n'
              '5) SIGMOID_CROSS_ENTROPY_LOSS\n'
              '6) MEAN_SQUARED_LOSS\n'
              '7) LIST_MLE_LOSS\n'
              '8) APPROX_NDCG_LOSS\n')

        selection = input()
        loss = None
        if selection == '1':
            loss = 'pairwise_hinge_loss'
        elif selection == '2':
            loss = 'pairwise_logistic_loss'
        elif selection == '3':
            loss = 'pairwise_soft_zero_one_loss'
        elif selection == '4':
            loss = 'softmax_loss'
        elif selection == '5':
            loss = 'sigmoid_cross_entropy_loss'
        elif selection == '6':
            loss = 'mean_squared_loss'
        elif selection == '7':
            loss = 'list_mle_loss'
        elif selection == '8':
            loss = 'approx_ndcg_loss'

        if _MODE == 'full':
            print('insert_num_train_step')
            num_train_steps = input()
        else:
            num_train_steps = None

        print('insert the group_size:')
        group_size = int(input())
        flags.DEFINE_integer("train_batch_size", train_batch_size, "The batch size for training.")
        flags.DEFINE_integer("num_train_steps", num_train_steps, "Number of steps for training.")
        flags.DEFINE_float("learning_rate", learning_rate, "Learning rate for optimizer.")
        flags.DEFINE_float("dropout_rate", dropout_rate, "The dropout rate before output layer.")
        flags.DEFINE_list("hidden_layer_dims", hidden_layer_dims,
                          "Sizes for hidden layers.")
        flags.DEFINE_integer("group_size", group_size, "Group size used in score function.")
        flags.DEFINE_string("loss", loss,
                            "The RankingLossKey for loss function.")
        FLAGS = flags.FLAGS
        tf.app.run()
