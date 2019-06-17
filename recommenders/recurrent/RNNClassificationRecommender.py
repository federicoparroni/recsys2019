import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
import numpy as np

from recommenders.RNN import RecurrentRecommender, mrr

from utils.dataset import SequenceDatasetForClassification

from utils.check_folder import check_folder
import out
import utils.submit as sub
from tqdm import tqdm


class RNNClassificationRecommender(RecurrentRecommender):
    """ Recurrent model for classification
        (the goal is to predict the index of the right clickout reference in the impressions list)
    """
    
    def __init__(self, dataset, input_shape, cell_type, num_recurrent_layers, num_recurrent_units, num_dense_layers,
                use_generator=False, validation_split=0.15, class_weights=None, sample_weights=None, output_size=25,
                metrics=['accuracy', mrr],
                optimizer='rmsprop', loss='categorical_crossentropy', batch_size=64,
                checkpoints_path=None, tensorboard_path=None):
        
        super().__init__(dataset=dataset, input_shape=input_shape, cell_type=cell_type, num_recurrent_layers=num_recurrent_layers,
                        num_recurrent_units=num_recurrent_units, num_dense_layers=num_dense_layers,
                        use_generator=use_generator, validation_split=validation_split, output_size=output_size,
                        loss=loss, optimizer=optimizer, class_weights=class_weights, sample_weights=sample_weights, metrics=metrics,
                        batch_size=batch_size,
                        checkpoints_path=checkpoints_path, tensorboard_path=tensorboard_path)
        
        self.name += '_class'
        

    def recommend_batch(self, target_indices):
        X, indices = self.dataset.load_Xtest()
        
        # predict the references
        predictions = self.model.predict(X)
        
        # take only the last index for each session (target row) and flatten
        #predictions = predictions.reshape((-1, predictions.shape[-1]))
        #indices = indices[:,-1].flatten()
        
        # take only the target predictions
        pred_df = pd.DataFrame(predictions)
        pred_df['orig_index'] = indices
        pred_df = pred_df.set_index('orig_index')
        predictions = pred_df.loc[target_indices]
        del pred_df

        assert len(predictions) == len(target_indices)

        full_df = data.full_df()

        result_predictions = []
        for index in tqdm(target_indices):
            # get the impressions of the clickout to predict
            impr = list(map(int, full_df.loc[index]['impressions'].split('|')))
            # build a list of (impression, score)
            prediction_impressions_distances = [ (impr[j], predictions.at[index,j]) for j in range(len(impr)) ]
            # order the list based on scores (greater is better)
            prediction_impressions_distances.sort(key=lambda tup: tup[1], reverse=True)
            # get only the impressions ids
            ordered_impressions = list(map(lambda x: x[0], prediction_impressions_distances))
            # append the couple (index, reranked impressions)
            result_predictions.append( (index, ordered_impressions) )

        print('prediction created !!!')

        return result_predictions

    def get_scores_batch(self, scores_type='test'):
        assert scores_type in ['train','test']

        if scores_type == 'test':
            X, indices = self.dataset.load_Xtest()
        else:
            X, indices = self.dataset.load_Xtrain(return_indices=True)

        predictions = self.model.predict(X)

        full_df = data.full_df()

        result_predictions = []
        for i,index in tqdm(enumerate(indices)):
            # get the impressions of the clickout to predict
            impr = list(map(int, full_df.loc[index]['impressions'].split('|')))
            scores = predictions[i]
            # append the triple (index, impressions, scores)
            result_predictions.append( (index, impr, scores) )

        return result_predictions

    def get_scores_cv(self, x, groups, test_indices):
        """ Return scores for a fold """
        x_val = x[test_indices]
        indices = x_val[:,:,0][:,self.dataset.rows_per_sample-1]

        predictions = self.model.predict(x_val[:,:,1:])

        # take the target rows
        res_df = data.full_df()[['user_id','session_id','impressions']].loc[indices].copy()
        res_df['impressions'] = res_df['impressions'].str.split('|')
        # add the scores as a new column
        res_df['scores'] = list(predictions)
        # trim the scores to the real number of impressions
        # (otherwise all rows have the fixed number of scores (25) )
        res_df['length'] = res_df['impressions'].str.len()
        res_df['scores'] = res_df.apply(lambda x: x['scores'][:x['length']], axis=1)
        res_df.drop('length', axis=1, inplace=True)

        # expand the df to have a row for each item_id - score
        res_df = pd.DataFrame({ col:np.repeat(res_df[col], res_df['scores'].str.len())
                    for col in res_df.columns.drop(['impressions','scores'])
        }).assign(**{
            'item_id': np.concatenate(res_df['impressions'].values),
            'score': np.concatenate(res_df['scores'].values),
        })

        return res_df


if __name__ == "__main__":
    import utils.menu as menu

    # build the weights array
    # weights = np.array([0.37738,0.10207,0.07179,0.05545,0.04711,0.03822,0.03215,0.02825,0.02574,
    #                     0.02289,0.02239,0.02041,0.01814,0.01619,0.01451,0.01306,0.01271,0.01156,
    #                     0.01174,0.01072,0.01018,0.00979,0.00858,0.00868,0.01029])
    # weights = 1/weights
    # wgt_sum = sum(weights)
    # weights = weights/wgt_sum
    # weights = dict([(i,w) for i,w in enumerate(weights)])
    # print(weights)

    def interactive_model(mode):
        cell_type = menu.single_choice('Choose a network architecture:', ['LSTM', 'GRU', 'default architecture'], [lambda: 'LSTM', lambda: 'GRU', lambda: 'auto'])
        print()
        if cell_type == 'auto':
            cell_type = 'GRU'
            rec_layers = 1
            dense_layers = 2
            units = 4
            weights = True
        else:
            rec_layers = int(input('Insert number of recurrent layers: '))
            units = int(input('Insert number of units per layer: '))
            dense_layers = int(input('Insert number of dense layers: '))
            weights = menu.yesno_choice('Do you want to use sample weights?', lambda: True, lambda: None)
            #tb_path = menu.yesno_choice('Do you want to enable Tensorboard?', lambda: 'recommenders/tensorboard', lambda: None)

        pad = menu.single_choice('Which dataset?', ['Padded 6','Padded 12'], [lambda: 6, lambda: 12])
        dataset = SequenceDatasetForClassification(f'dataset/preprocessed/cluster_recurrent/{mode}/dataset_classification_p{pad}')
        
        if weights is not None:
            weights = dataset.get_sample_weights()
        
        model = RNNClassificationRecommender(dataset, use_generator=False, cell_type=cell_type,
                                            input_shape=(dataset.rows_per_sample, 168),
                                            num_recurrent_layers=rec_layers, num_recurrent_units=units, optimizer='adam',
                                            num_dense_layers=dense_layers,
                                            #class_weights=weights
                                            sample_weights=weights
                                            )

        return model

    def train():
        mode = menu.mode_selection()
        # fit the model
        model = interactive_model(mode)
        model.fit(epochs=10000, early_stopping_patience=25, early_stopping_on='val_mrr', mode='max')
        print('\nFit completed!')

        # recommend
        target_indices = data.target_indices(mode, 'cluster_recurrent')
        print('Recommending...')
        recommendations = model.recommend_batch(target_indices)
        print('Recommendation count: ', len(recommendations))
        mrr = model.compute_MRR(recommendations)

        model.save(folderpath='saved_models/', suffix='_{}'.format(round(mrr, 5)).replace('.','') )

    def submission():
        mode = 'full'
        model = interactive_model(mode)
        sub_suffix = input('Insert submission suffix: ')

        checkpoint_path = menu.checkpoint_selection(checkpoints_dir='saved_models')

        print('Loading {}...'.format(checkpoint_path), end='\r', flush=True)
        model.load(checkpoint_path)
        print('Done!',  flush=True)

        # recommend
        target_indices = data.target_indices(mode, 'cluster_recurrent')
        print('Recommending...')
        recommendations = model.recommend_batch(target_indices)

        # create and send sub
        sub_name = f'{model.name}_{sub_suffix}'
        sub_path = out.create_sub(recommendations, submission_name=sub_name)
        print('Done')
        sub.send(sub_path, username='federico.parroni@live.it', password='siamoi3piùcarichi')

    def scores():
        mode = 'full'
        model = interactive_model(mode)

        checkpoint_path = menu.checkpoint_selection(checkpoints_dir='saved_models')

        print('Loading {}...'.format(checkpoint_path), end='\r', flush=True)
        model.load(checkpoint_path)
        print('Done!',  flush=True)

        # get scores for train and test and save them
        scores_folder = 'scores'
        check_folder(scores_folder)

        for st in ['train', 'test']:
            print('Building scores for {} {}...'.format(st, mode))
            scores = model.get_scores_batch(scores_type=st)
            
            print('Saving scores for {} {}...'.format(st, mode))
            scores_filename = '{}_scores_{}'.format(model.name, st)
            np.save(os.path.join(scores_folder, scores_filename), np.array(scores))
            

    
    activity = menu.single_choice('What do you want to do?', ['Train', 'Submission', 'Scores'], [train, submission, scores])
        
