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

  def __init__(self, min_mrr_start, loss, dataset_name, save_path, test_x, test_y, mode, name='best_checkpoints', checkpoints_to_keep=5, score_metric='Loss/total_loss',
               compare_fn=lambda x,y: x.score < y.score, sort_key_fn=lambda x: x.score, sort_reverse=False):
    self.checkpoints = []
    self.checkpoints_to_keep = checkpoints_to_keep
    self.compare_fn = compare_fn
    self.name = name
    self.score_metric = score_metric
    self.sort_key_fn = sort_key_fn
    self.sort_reverse = sort_reverse

    self.mode = mode
    self.test_x = test_x
    self.test_y = test_y
    self.save_path = save_path
    self.dataset_name = dataset_name
    self.loss = loss
    self.min_mrr = min_mrr_start
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
    HERA.send_message(f'TFRANKING mrr is: {eval_result[self.score_metric]}')
    return float(eval_result[self.score_metric])

  def _shouldKeep(self, checkpoint):
    return len(self.checkpoints) < self.checkpoints_to_keep or self.compare_fn(checkpoint, self.checkpoints[-1])

  def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):

    def batch_inputs(features, labels, batch_size):
      dataset = tf.data.Dataset.from_tensor_slices((features, labels))
      return dataset.batch(batch_size)

    def create_sub(estimator, checkpoint_path, eval_result, batch_size=64, patience=0.01):
      #TODO!!!!!!!!!!!!!!!!!!! SELECT FULL
      if self.mode == 'small':
        # create a sub only if the MMR is > 0.65
        if eval_result[self.score_metric]>self.min_mrr+patience:
          # set as new threshold the new mrr
          self.min_mrr = eval_result[self.score_metric]

          pred = np.array(list(estimator.predict(lambda: batch_inputs(self.test_x, self.test_y, batch_size))))
          np.save(self.save_path, pred)
          HERA.send_message(f'EXPORTING A SUB... {eval_result}')
          model = TensorflowRankig(mode='full', cluster='no_cluster', dataset_name=self.dataset_name)
          model.name = f'tf_ranking_{self.loss}_{eval_result}'
          model.run()

    self._log('export checkpoint {}'.format(checkpoint_path))

    score = self._score(eval_result)
    checkpoint = Checkpoint(path=checkpoint_path, score=score)

    if self._shouldKeep(checkpoint):
      self._keepCheckpoint(checkpoint)

      create_sub(estimator, checkpoint_path, eval_result)

      self._pruneCheckpoints(checkpoint)
    else:
      self._log('skipping checkpoint {}'.format(checkpoint.path))