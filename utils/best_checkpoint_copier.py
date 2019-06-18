import glob
import os
import shutil
import tensorflow as tf
import numpy as np
import utils.telegram_bot as HERA
from recommenders.tf_ranking import TensorflowRankig

class Checkpoint(object):
  dir = None
  file = None
  score = None
  path = None

  def __init__(self, path, score):
    self.dir = os.path.dirname(path)
    self.file = os.path.basename(path)
    self.score = score
    self.path = path


class BestCheckpointCopier(tf.estimator.Exporter):
  checkpoints = None
  checkpoints_to_keep = None
  compare_fn = None
  name = None
  score_metric = None
  sort_key_fn = None
  sort_reverse = None

  def __init__(self, min_mrr_start, loss, dataset_name, save_path, x, y, test_x, test_y, params,
               mode, name='best_checkpoints', checkpoints_to_keep=1, score_metric='Loss/total_loss',
               compare_fn=lambda x,y: x.score > y.score, sort_key_fn=lambda x: x.score, sort_reverse=False):
    self.checkpoints = []
    self.checkpoints_to_keep = checkpoints_to_keep
    self.compare_fn = compare_fn
    self.name = name
    self.score_metric = score_metric
    self.sort_key_fn = sort_key_fn
    self.sort_reverse = sort_reverse

    self.mode = mode

    self.x = x
    self.y = y
    self.test_x = test_x
    self.test_y = test_y
    self.save_path = save_path

    self.dataset_name = dataset_name
    self.loss = loss
    self.min_mrr = min_mrr_start
    self.params = params
    super(BestCheckpointCopier, self).__init__()

  def _copyCheckpoint(self, checkpoint):
    desination_dir = self._destinationDir(checkpoint)
    os.makedirs(desination_dir, exist_ok=True)

    for file in glob.glob(r'{}*'.format(checkpoint.path)):
      self._log('copying {} to {}'.format(file, desination_dir))
      shutil.copy(file, desination_dir)

  def _destinationDir(self, checkpoint):
    return os.path.join(checkpoint.dir, self.name)

  def _keepCheckpoint(self, checkpoint):
    self._log('keeping checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

    self.checkpoints.append(checkpoint)
    self.checkpoints = sorted(self.checkpoints, key=self.sort_key_fn, reverse=self.sort_reverse)

    self._copyCheckpoint(checkpoint)

  def _log(self, statement):
    tf.logging.info('[{}] {}'.format(self.__class__.__name__, statement))

  def _pruneCheckpoints(self, checkpoint):
    destination_dir = self._destinationDir(checkpoint)

    for checkpoint in self.checkpoints[self.checkpoints_to_keep:]:
      self._log('removing old checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

      old_checkpoint_path = os.path.join(destination_dir, checkpoint.file)
      for file in glob.glob(r'{}*'.format(old_checkpoint_path)):
        self._log('removing old checkpoint file {}'.format(file))
        os.remove(file)
    self.checkpoints = self.checkpoints[0:self.checkpoints_to_keep]

  def _score(self, eval_result):
    return float(eval_result)

  def _shouldKeep(self, checkpoint):
    return len(self.checkpoints) < self.checkpoints_to_keep or checkpoint.score>self.checkpoints[-1].score

  def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):

    def batch_inputs(features, labels, batch_size):
      dataset = tf.data.Dataset.from_tensor_slices((features, labels))
      return dataset.batch(batch_size)

    def create_sub(estimator, checkpoint_path, eval_result, batch_size=128, patience=0.001):
      # now works also for local and small it will create a sub
      # create a sub only if the MMR is > 0.65
      if self.mode == 'local':
        eval_result_f = eval_result['metric/mrr']
        global_step = eval_result['global_step']
        if eval_result_f>self.min_mrr+patience:
          # set as new threshold the new mrr
          self.min_mrr = eval_result_f

          # predict the test...
          pred = np.array(list(estimator.predict(lambda: batch_inputs(self.test_x, self.test_y, batch_size))))
          pred_train = np.array(list(estimator.predict(lambda: batch_inputs(self.x, self.y, batch_size))))

          pred_name_train = 'train_predictions_{}_learning_rate_{}_train_batch_size_{}_hidden_layers_dim_{}_num_train_steps_{}' \
          '_dropout_{}_global_steps_{}_{}_mrr_{}'.format(self.params['loss'], self.params['learning_rate'], self.params['train_batch_size'], self.params['hidden_layer_dims'],
          self.params['num_train_steps'], self.params['dropout_rate'], global_step, self.params['group_size'], eval_result_f)

          pred_name = 'predictions_{}_learning_rate_{}_train_batch_size_{}_hidden_layers_dim_{}_num_train_steps_{}' \
          '_dropout_{}_global_steps_{}_{}_mrr_{}'.format(self.params['loss'], self.params['learning_rate'], self.params['train_batch_size'], self.params['hidden_layer_dims'],
          self.params['num_train_steps'], self.params['dropout_rate'], global_step, self.params['group_size'], eval_result_f)

          np.save(f'{self.save_path}/{pred_name_train}', pred_train)
          np.save(f'{self.save_path}/{pred_name}', pred)

          for name in [pred_name, pred_name_train]:
            HERA.send_message(f'EXPORTING A SUB... {eval_result_f} mode:{self.mode}, name:{name}')
            model = TensorflowRankig(mode=self.mode, cluster='no_cluster', dataset_name=self.dataset_name,
                                     pred_name=name)
            model.name = f'tf_ranking_{name}'
            model.run()
            HERA.send_message(f'EXPORTED... {eval_result_f} mode:{self.mode}, name:{name}')

    self._log('export checkpoint {}'.format(checkpoint_path))
    step = eval_result['global_step']
    score = eval_result['metric/mrr']
    checkpoint = Checkpoint(path=checkpoint_path, score=score)


    HERA.send_message('mode: {}\n step:{}\nTFRANKING mrr is: {}\n dropout:{}\n'
                      'learning_rate:{}\n train_batch_size:{}\n'
                      'hidden_layer_dims:{}\n loss:{}\n group_size:{}'.format(self.mode, step, score, self.params['dropout_rate'], self.params['learning_rate'], self.params['train_batch_size'],
    self.params['hidden_layer_dims'], self.params['loss'], self.params['group_size']))
    if self._shouldKeep(checkpoint):
      self._keepCheckpoint(checkpoint)

      create_sub(estimator, checkpoint_path, eval_result)

      self._pruneCheckpoints(checkpoint)
    else:
      self._log('skipping checkpoint {}'.format(checkpoint.path))