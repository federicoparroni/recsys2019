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
import utils.menu as menu
import sys
from tqdm import tqdm
import pandas as pd

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
    # check whether the values of a list are all identical
    _RESULT = True
    current_el = list[0]
    for i in range(1,len(list),1):
        if list[i] != current_el:
            _RESULT = False
            break
    return _RESULT

def example_feature_columns():
  """Returns the example feature columns."""
  global flags_dict
  feature_names = ["{}".format(i + 1) for i in range(flags_dict['num_features'])]
  return {
      name: tf.feature_column.numeric_column(
          name, shape=(1,), default_value=0.0) for name in feature_names
  }

def load_data(path, list_size):
  """Returns features and labels in numpy.array."""
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
    global flags_dict
    with tf.name_scope("input_layer"):
      """
      names = sorted(example_feature_columns())
      names.remove('28')
      group_input = [
          tf.layers.flatten(group_features[name])
          for name in names
      ]
      """
      per_ex_features = tf.concat([tf.layers.flatten(group_features[name]) for name in flags_dict['train_per_example_features_id']],1)
      context_features = tf.concat([tf.layers.flatten(unused_context_features[name]) for name in flags_dict['train_context_features_id']], 1)

      input_layer = tf.concat([per_ex_features, context_features], axis=1)
      tf.summary.scalar("input_sparsity", tf.nn.zero_fraction(input_layer))
      tf.summary.scalar("input_max", tf.reduce_max(input_layer))

      tf.summary.scalar("input_min", tf.reduce_min(input_layer))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = tf.layers.batch_normalization(input_layer, training=is_training)
    for i, layer_width in enumerate(int(d) for d in flags_dict['hidden_layer_dims']):
      cur_layer = tf.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.layers.batch_normalization(cur_layer, training=is_training)
      cur_layer = tf.nn.relu(cur_layer)
      tf.summary.scalar("fully_connected_{}_sparsity".format(i),
                        tf.nn.zero_fraction(cur_layer))
    cur_layer = tf.layers.dropout(
    cur_layer, rate=flags_dict['dropout_rate'], training=is_training)
    logits = tf.layers.dense(cur_layer, units=flags_dict['group_size'])
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
      _features, _labels = load_data(flags_dict['train_path'], flags_dict['list_size'])
  if (_features_vali is None) or (_labels_vali is None):
      print('caching data test')
      _features_vali, _labels_vali = load_data(flags_dict['vali_path'],
                                                flags_dict['list_size'])

  train_input_fn, train_hook = get_train_inputs(_features, _labels,
                                                flags_dict['train_batch_size'])

  vali_input_fn, vali_hook = get_batches(_features_vali, _labels_vali,
                                                flags_dict['train_batch_size'])

  def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=flags_dict['learning_rate'],
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
      dataset_name=flags_dict['dataset_name'],
      save_path=flags_dict['save_path'],
      #save_path=f'dataset/preprocessed/tf_ranking/{_CLUSTER}/full/{_DATASET_NAME}/predictions',
      x=_features,
      y=_labels,
      test_x=_features_vali,
      test_y=_labels_vali,
      mode=flags_dict['mode'],
      loss=flags_dict['loss'],
      min_mrr_start=flags_dict['min_mrr_start'],
      params=flags_dict
      )

  if flags_dict['loss'] == 'list_mle_loss':
      lambda_weight = tfr.losses.create_p_list_mle_lambda_weight(list_size=25)
  elif flags_dict['loss'] == 'approx_ndcg_loss':
      lambda_weight = tfr.losses.create_ndcg_lambda_weight()
  else:
      lambda_weight = tfr.losses.create_reciprocal_rank_lambda_weight()


  ranking_head = tfr.head.create_ranking_head(
      loss_fn=tfr.losses.make_loss_fn(flags_dict['loss'], lambda_weight=lambda_weight),
      eval_metric_fns=get_eval_metric_fns(),
      train_op_fn=_train_op_fn)

  #weights_feature_name=FLAGS.weights_feature_number
  #lambda_weight=tfr.losses.create_reciprocal_rank_lambda_weight(smooth_fraction=0.5)

  estimator = tf.estimator.Estimator(
      model_fn=tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          group_size=flags_dict['group_size'],
          transform_fn=tfr.feature.make_identity_transform_fn(flags_dict['train_context_features_id']),
          #tfr.feature.make_identity_transform_fn(['28'])
          ranking_head=ranking_head),
    config=tf.estimator.RunConfig(
      flags_dict['output_dir'], save_checkpoints_steps=flags_dict['save_checkpoints_steps']))


  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      hooks=[train_hook],
      max_steps=flags_dict['num_train_steps'])
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
    features, labels = load_data(flags_dict['train_path'], flags_dict['list_size'])
    train_input_fn, train_hook = get_train_inputs(features, labels,
                                                  flags_dict['train_batch_size'])
    features_test, labels_test = load_data(flags_dict['test_path'],
                                                  flags_dict['list_size'])

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=flags_dict['learning_rate'],
            optimizer="Adagrad")

    if flags_dict['loss'] == 'list_mle_loss':
        lambda_weight = tfr.losses.create_p_list_mle_lambda_weight(list_size=25)
    elif flags_dict['loss'] == 'approx_ndcg_loss':
        lambda_weight = tfr.losses.create_ndcg_lambda_weight(topn=25)
    else:
        lambda_weight = tfr.losses.create_reciprocal_rank_lambda_weight(topn=25)
    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(flags_dict['loss'], lambda_weight=lambda_weight),
        eval_metric_fns=get_eval_metric_fns(),
        train_op_fn=_train_op_fn)
    # tfr.losses.create_p_list_mle_lambda_weight(25)
    # lambda_weight=tfr.losses.create_reciprocal_rank_lambda_weight()

    estimator = tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=flags_dict['group_size'],
            transform_fn=tfr.feature.make_identity_transform_fn(flags_dict['train_context_features_id']),
            ranking_head=ranking_head))

    estimator.train(train_input_fn, hooks=[train_hook], steps=flags_dict['num_train_steps'])

    # predict also for the train to get the scores for the staking
    pred_train = np.array(list(estimator.predict(lambda: batch_inputs(features, labels, 128))))
    pred = np.array(list(estimator.predict(lambda: batch_inputs(features_test, labels_test, 128))))




    pred_name_train='train_predictions_{}_learning_rate_{}_train_batch_size_{}_' \
        'hidden_layers_dim_{}_num_train_steps_{}_dropout_{}_group_size_{}'.format(flags_dict['loss'],
                                                                                  flags_dict['learning_rate'],
                                                                                  flags_dict['train_batch_size'],
                                                                                  flags_dict['hidden_layer_dims'],
                                                                                  flags_dict['num_train_steps'],
                                                                                  flags_dict['dropout_rate'],
                                                                                  flags_dict['group_size'])

    pred_name ='predictions_{}_learning_rate_{}_train_batch_size_{}_' \
        'hidden_layers_dim_{}_num_train_steps_{}_dropout_{}_group_size_{}'.format(flags_dict['loss'],
                                                                                  flags_dict['learning_rate'],
                                                                                  flags_dict['train_batch_size'],
                                                                                  flags_dict['hidden_layer_dims'],
                                                                                  flags_dict['num_train_steps'],
                                                                                  flags_dict['dropout_rate'],
                                                                                  flags_dict['group_size'])
    np.save('{}/{}'.format(flags_dict['save_path'], pred_name), pred)
    np.save('{}/{}'.format(flags_dict['save_path'], pred_name_train), pred_train)

    for name in [pred_name, pred_name_train]:
        HERA.send_message('EXPORTING A SUB... mode:{}, name:{}'.format(flags_dict['mode'], name))
        model = TensorflowRankig(mode=flags_dict['mode'], cluster='no_cluster', dataset_name=flags_dict['dataset_name'],
                                 pred_name=name)
        model.name = f'tf_ranking_{name}'
        model.run()
        HERA.send_message('EXPORTED... mode:{}, name:{}'.format(flags_dict['mode'], name))

if __name__ == '__main__':
    global flags_dict
    flags_dict = {}

    # let the user insert the mode cluster and dataset name
    _MODE = menu.mode_selection()
    _CLUSTER = menu.cluster_selection()
    _DATASET_NAME = input('insert dataset name: \n')

    _BASE_PATH = f'dataset/preprocessed/tf_ranking/{_CLUSTER}/{_MODE}/{_DATASET_NAME}'
    _TRAIN_PATH = f'{_BASE_PATH}/train.hdf'
    _TEST_PATH = f'{_BASE_PATH}/test.hdf'
    _VALI_PATH = f'{_BASE_PATH}/vali.hdf'

    min_mrr = input('insert the min_MRR from which export the sub: \n')

    # lets load the context features id
    context_features_id = list(np.load(f'{_BASE_PATH}/context_features_id.npy'))
    print(f'context features id are: {context_features_id}')
    flags_dict['context_features_id']=context_features_id

    # retrieve the number of features
    with open(f'{_BASE_PATH}/features_num.txt') as f:
        num_features = int(f.readline())
        print(f'num_features is: {num_features}')

    # let user insert save_check_steps
    save_check_steps = int(input('insert save_check_steps: \n'))

    # update the flags dict
    flags_dict['save_path'] = _BASE_PATH
    flags_dict['train_path'] = _TRAIN_PATH
    flags_dict['vali_path'] = _VALI_PATH
    flags_dict['test_path'] =_TEST_PATH
    flags_dict['mode']= _MODE
    flags_dict['dataset_name'] = _DATASET_NAME
    flags_dict['num_features'] = num_features
    flags_dict['list_size'] = 25
    flags_dict['save_checkpoints_steps'] =save_check_steps
    flags_dict['min_mrr_start'] = min_mrr

    cf.check_folder(f'{_BASE_PATH}/output_dir_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}')
    _OUTPUT_DIR = f'{_BASE_PATH}/output_dir_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}'

    flags_dict['output_dir'] = _OUTPUT_DIR
    # let the user choose the params
    train_batch_size = int(input('insert train batch size'))
    learning_rate = float(input('insert learning rate'))
    dropout_rate = float(input('insert dropout rate'))
    hidden_layer_dims = input('insert hidden layer dims as numbers separeted by spaces').split(' ')
    loss = menu.single_choice('select the loss', ['pairwise_hinge_loss',
                                                  'pairwise_logistic_loss',
                                                  'pairwise_soft_zero_one_loss',
                                                  'softmax_loss',
                                                  'sigmoid_cross_entropy_loss',
                                                  'mean_squared_loss',
                                                  'list_mle_loss',
                                                  'approx_ndcg_loss'])

    if _MODE == 'full':
        num_train_steps = input('insert_num_train_step')
    else:
        num_train_steps = None

    group_size = int(input('insert the group_size:\n'))

    #update flag dict
    flags_dict['train_batch_size'] = train_batch_size
    flags_dict['num_train_steps'] = num_train_steps
    flags_dict['learning_rate'] = learning_rate
    flags_dict['dropout_rate'] = dropout_rate
    flags_dict['hidden_layer_dims'] = hidden_layer_dims
    flags_dict['group_size'] = group_size
    flags_dict['loss'] = loss

    if _MODE == 'full':
        train_and_test()
    else:
        train_and_eval()



