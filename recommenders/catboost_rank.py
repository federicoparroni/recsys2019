import math

import data
from recommenders.recommender_base import RecommenderBase
from tqdm import tqdm
from catboost import CatBoost, Pool
from copy import deepcopy
import pickle
import pandas as pd
tqdm.pandas()

class CatboostRanker(RecommenderBase):
    """
    Catboost by Yandex for ranking purposes
    Adapted from tutorial:
    https://github.com/catboost/tutorials/blob/master/ranking/ranking_tutorial.ipynb
    Custom_metric is @1 for maximizing first result as good
    """

    def __init__(self, mode, cluster='no_cluster', learning_rate=0.15, iterations=200, max_depth=8, reg_lambda=3,
                 colsample_bylevel=1,
                 custom_metric='AverageGain:top=1', algo='xgboost', verbose=False, include_test=False, file_to_load=None,
                 file_to_store=None, limit_trees=False, features_to_one_hot = None):
        """
        :param mode:
        :param cluster:
        :param iterations: number of trees to use
        :param include_test: True if use test for allow visual early stopping
        :param custom_metric: Metric to evaluate during training
        :param verbose: True if writing a log of training
        :param file_to_load: specify the path of an existing model to use without training a new one
        :param file_to_store: specify the path where the model will be stored
        :param limit_trees: limit trees to use whenever an existing model is being used
        """
        name = 'catboost_rank'
        super(CatboostRanker, self).__init__(
            name=name, mode=mode, cluster=cluster)

        self.target_indices = data.target_indices(mode=mode, cluster=cluster)
        self.features_to_drop = []
        self.include_test = include_test

        self.default_parameters = {
            'iterations': math.ceil(iterations),
            'custom_metric': custom_metric,
            'verbose': verbose,
            'random_seed': 0,
            'learning_rate': learning_rate,
            'max_depth': math.ceil(max_depth),
            'colsample_bylevel': math.ceil(colsample_bylevel),
            'reg_lambda': math.ceil(reg_lambda),
            'loss_function': 'QuerySoftMax',
            'train_dir': 'QuerySoftMax'
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {'iterations': (10, 1000),
                                     'max_depth': (3, 8),
                                     'learning_rate': (0.01, 0.2),
                                     'reg_lambda': (1, 5),
                                     }

        self.fixed_params_dict = {
            'mode': mode,
            'cluster': cluster,
            'colsample_bylevel': 1
        }

        self.limit_trees = limit_trees
        self.file_to_load = file_to_load
        self.file_to_store = file_to_store
        self.features_to_one_hot = features_to_one_hot
        self.algo = algo

        self.categorical_features = None

    def fit_model(self, additional_params=None, train_pool=None, test_pool=None):
        parameters = deepcopy(self.default_parameters)

        if additional_params is not None:
            parameters.update(additional_params)

        model = CatBoost(parameters)
        model.fit(train_pool, eval_set=test_pool, plot=True)

        return model

    def fit(self):

        if self.file_to_load is not None:
            # --- To load model ---
            self.ctb = pickle.load(open(self.file_to_load, 'rb'))
            print("Model loaded")
            return

        print('Start training the model...')
        train_df = data.classification_train_df(mode=self.mode, cluster=self.cluster, sparse=False, algo=self.algo)

        if len(self.features_to_drop)>0:
            train_df = train_df.reindex(
                'user_id,session_id,label,timing_last_action_before_clk,search_for_poi_distance_from_last_clickout,search_for_poi_distance_from_first_action,change_sort_order_distance_from_last_clickout,change_sort_order_distance_from_first_action,impression_position_wrt_last_interaction,impression_position_wrt_second_last_interaction,times_impression_appeared,time_per_step,time_elapsed_from_last_time_impression_appeared,tablet,steps_from_last_time_impression_appeared,sorted by default,sort by rating only,sort by rating and recommended,sort by price only,sort by price and recommended,sort by our recommendations,sort by interaction sort button,sort by distance only,sort by distance and recommended,session_length_in_time,session_length_in_step,search_for_item_session_ref_this_impr,search_for_item_session_ref_not_in_impr,price_position,price,pos_last_interaction_in_impressions,popularity,mobile,last_time_reference_did_not_appeared,last_time_impression_appeared_as_search_for_item,last_time_impression_appeared_as_interaction_item_rating,last_time_impression_appeared_as_interaction_item_info,last_time_impression_appeared_as_interaction_item_image,last_time_impression_appeared_as_interaction_item_deals,last_time_impression_appeared_as_clickout_item,interaction_item_rating_session_ref_this_impr,interaction_item_rating_session_ref_not_in_impr,interaction_item_info_session_ref_this_impr,interaction_item_info_session_ref_not_in_impr,interaction_item_image_session_ref_this_impr,interaction_item_image_session_ref_not_in_impr,interaction_item_deals_session_ref_this_impr,interaction_item_deals_session_ref_not_in_impr,impression_position,frenzy_factor,desktop,clickout_item_session_ref_this_impr,clickout_item_session_ref_not_in_impr,avg_price_interacted_item,avg_pos_interacted_items_in_impressions,average_price_position,accomodation feature WiFi (Rooms),accomodation feature WiFi (Public Areas),accomodation feature Wheelchair Accessible,accomodation feature Water Slide,accomodation feature Washing Machine,accomodation feature Volleyball,accomodation feature Very Good Rating,accomodation feature Towels,accomodation feature Theme Hotel,accomodation feature Terrace (Hotel),accomodation feature Tennis Court (Indoor),accomodation feature Tennis Court,accomodation feature Television,accomodation feature Teleprinter,accomodation feature Telephone,accomodation feature Table Tennis,accomodation feature Szep Kartya,accomodation feature Swimming Pool (Outdoor),accomodation feature Swimming Pool (Indoor),accomodation feature Swimming Pool (Combined Filter),accomodation feature Swimming Pool (Bar),accomodation feature Surfing,accomodation feature Sun Umbrellas,accomodation feature Steam Room,accomodation feature Spa Hotel,accomodation feature Spa (Wellness Facility),accomodation feature Solarium,accomodation feature Skiing,accomodation feature Ski Resort,accomodation feature Sitting Area (Rooms),accomodation feature Singles,accomodation feature Shower,accomodation feature Shooting Sports,accomodation feature Serviced Apartment,accomodation feature Senior Travellers,accomodation feature Self Catering,accomodation feature Sauna,accomodation feature Satisfactory Rating,accomodation feature Satellite TV,accomodation feature Sailing,accomodation feature Safe (Rooms),accomodation feature Safe (Hotel),accomodation feature Room Service (24/7),accomodation feature Room Service,accomodation feature Romantic,accomodation feature Restaurant,accomodation feature Resort,accomodation feature Reception (24/7),accomodation feature Radio,accomodation feature Pousada (BR),accomodation feature Porter,accomodation feature Pool Table,accomodation feature Playground,accomodation feature Pet Friendly,accomodation feature Organised Activities,accomodation feature Openable Windows,accomodation feature On-Site Boutique Shopping,accomodation feature Non-Smoking Rooms,accomodation feature Nightclub,accomodation feature Motel,accomodation feature Minigolf,accomodation feature Microwave,accomodation feature Massage,accomodation feature Luxury Hotel,accomodation feature Lift,accomodation feature Laundry Service,accomodation feature Large Groups,accomodation feature Kosher Food,accomodation feature Kids\' Club,accomodation feature Jacuzzi (Hotel),accomodation feature Ironing Board,accomodation feature Hypoallergenic Rooms,accomodation feature Hypoallergenic Bedding,accomodation feature Hydrotherapy,accomodation feature House / Apartment,accomodation feature Hotel Bar,accomodation feature Hotel,accomodation feature Hot Stone Massage,accomodation feature Hostel,accomodation feature Hostal (ES),accomodation feature Horse Riding,accomodation feature Honeymoon,accomodation feature Hiking Trail,accomodation feature Health Retreat,accomodation feature Hammam,accomodation feature Halal Food,accomodation feature Hairdryer,accomodation feature Hairdresser,accomodation feature Gym,accomodation feature Guest House,accomodation feature Good Rating,accomodation feature Golf Course,accomodation feature Gay-friendly,accomodation feature From 4 Stars,accomodation feature From 3 Stars,accomodation feature From 2 Stars,accomodation feature Fridge,accomodation feature Free WiFi (Rooms),accomodation feature Free WiFi (Public Areas),accomodation feature Free WiFi (Combined),accomodation feature Flatscreen TV,accomodation feature Fitness,accomodation feature Farmstay,accomodation feature Fan,accomodation feature Family Friendly,accomodation feature Express Check-In / Check-Out,accomodation feature Excellent Rating,accomodation feature Electric Kettle,accomodation feature Eco-Friendly hotel,accomodation feature Doctor On-Site,accomodation feature Diving,accomodation feature Direct beach access,accomodation feature Desk,accomodation feature Design Hotel,accomodation feature Deck Chairs,accomodation feature Country Hotel,accomodation feature Cot,accomodation feature Cosmetic Mirror,accomodation feature Convention Hotel,accomodation feature Convenience Store,accomodation feature Conference Rooms,accomodation feature Concierge,accomodation feature Computer with Internet,accomodation feature Club Hotel,accomodation feature Childcare,accomodation feature Central Heating,accomodation feature Casino (Hotel),accomodation feature Casa Rural (ES),accomodation feature Car Park,accomodation feature Camping Site,accomodation feature Cable TV,accomodation feature Business Hotel,accomodation feature Business Centre,accomodation feature Bungalows,accomodation feature Bowling,accomodation feature Boutique Hotel,accomodation feature Body Treatments,accomodation feature Boat Rental,accomodation feature Bike Rental,accomodation feature Bed & Breakfast,accomodation feature Beauty Salon,accomodation feature Beach Bar,accomodation feature Beach,accomodation feature Bathtub,accomodation feature Balcony,accomodation feature All Inclusive (Upon Inquiry),accomodation feature Airport Shuttle,accomodation feature Airport Hotel,accomodation feature Air Conditioning,accomodation feature Adults Only,accomodation feature Accessible Parking,accomodation feature Accessible Hotel,accomodation feature 5 Star,accomodation feature 4 Star,accomodation feature 3 Star,accomodation feature 2 Star,accomodation feature 1 Star,Wheelchair Accessible filter active when clickout,Water Slide filter active when clickout,Washing Machine filter active when clickout,Very Good Rating filter active when clickout,Top Deals filter active when clickout,Today filter active when clickout,Terrace (Hotel) filter active when clickout,Tennis Court filter active when clickout,Television filter active when clickout,Teleprinter filter active when clickout,Telephone filter active when clickout,Swimming Pool (Outdoor) filter active when clickout,Swimming Pool (Indoor) filter active when clickout,Swimming Pool (Combined Filter) filter active when clickout,Surfing filter active when clickout,Sun Umbrellas filter active when clickout,Steam Room filter active when clickout,Spa Hotel filter active when clickout,Spa (Wellness Facility) filter active when clickout,Sort by Price filter active when clickout,Sort By Rating filter active when clickout,Sort By Popularity filter active when clickout,Sort By Distance filter active when clickout,Skiing filter active when clickout,Ski Resort filter active when clickout,Singles filter active when clickout,Shower filter active when clickout,Shooting Sports filter active when clickout,Serviced Apartment filter active when clickout,Senior Travellers filter active when clickout,Self Catering filter active when clickout,Sauna filter active when clickout,Satisfactory Rating filter active when clickout,Satellite TV filter active when clickout,Safe (Hotel) filter active when clickout,Room Service filter active when clickout,Room Service (24/7) filter active when clickout,Romantic filter active when clickout,Restaurant filter active when clickout,Resort filter active when clickout,Reception (24/7) filter active when clickout,Pousada (BR) filter active when clickout,Porter filter active when clickout,Playground filter active when clickout,Pet Friendly filter active when clickout,Organised Activities filter active when clickout,Openable Windows filter active when clickout,On-Site Boutique Shopping filter active when clickout,Non-Smoking Rooms filter active when clickout,Nightclub filter active when clickout,Motel filter active when clickout,Minigolf filter active when clickout,Microwave filter active when clickout,Massage filter active when clickout,Luxury Hotel filter active when clickout,Lift filter active when clickout,Last Minute filter active when clickout,Kosher Food filter active when clickout,Kitchen filter active when clickout,Kids\' Club filter active when clickout,Jacuzzi (Hotel) filter active when clickout,Internet (Rooms) filter active when clickout,Hypoallergenic Bedding filter active when clickout,House / Apartment filter active when clickout,Hotel filter active when clickout,Hotel Chain filter active when clickout,Hotel Bar filter active when clickout,Hostel filter active when clickout,Hostal (ES) filter active when clickout,Honeymoon filter active when clickout,Holiday filter active when clickout,Health Retreat filter active when clickout,Hammam filter active when clickout,Halal Food filter active when clickout,Hairdryer filter active when clickout,Gym filter active when clickout,Guest House filter active when clickout,Good Rating filter active when clickout,Golf Course filter active when clickout,Gay Friendly filter active when clickout,From 2 Stars filter active when clickout,Fridge filter active when clickout,Free WiFi (Rooms) filter active when clickout,Free WiFi (Public Areas) filter active when clickout,Free WiFi (Combined) filter active when clickout,Focus on Rating filter active when clickout,Focus on Distance filter active when clickout,Flatscreen TV filter active when clickout,Fitness filter active when clickout,Farmstay filter active when clickout,Family Friendly filter active when clickout,Express Check-In / Check-Out filter active when clickout,Excellent Rating filter active when clickout,Diving filter active when clickout,Disneyland Paris filter active when clickout,Direct beach access filter active when clickout,Deck Chairs filter active when clickout,Deals + Beach (NL/BE) filter active when clickout,Deals + Beach (MX) filter active when clickout,Country Hotel filter active when clickout,Cot filter active when clickout,Convenience Store filter active when clickout,Conference Rooms filter active when clickout,Childcare filter active when clickout,Cheap filter active when clickout,Central Heating filter active when clickout,Casino (Hotel) filter active when clickout,Casa Rural (ES) filter active when clickout,Car Park filter active when clickout,Camping Site filter active when clickout,Business Hotel filter active when clickout,Business Centre filter active when clickout,Bungalows filter active when clickout,Breakfast Included filter active when clickout,Boutique Hotel filter active when clickout,Best Value filter active when clickout,Best Rates filter active when clickout,Bed & Breakfast filter active when clickout,Beach filter active when clickout,Beach Bar filter active when clickout,Bathtub filter active when clickout,Balcony filter active when clickout,All Inclusive (Upon Inquiry) filter active when clickout,Airport Shuttle filter active when clickout,Airport Hotel filter active when clickout,Air Conditioning filter active when clickout,Adults Only filter active when clickout,Accessible Parking filter active when clickout,Accessible Hotel filter active when clickout,5 Star filter active when clickout,4 Star filter active when clickout,3 Star filter active when clickout,2 Star filter active when clickout,2 Nights filter active when clickout,1 Star filter active when clickout'.split(
                    ',')
                , axis=1)
            #train_df.drop(self.features_to_drop, axis=1, inplace=True)

        train_df = train_df.drop(['times_doubleclickout_on_item'], axis=1)

        #train_df.drop(['avg_price_interacted_item','average_price_position', 'avg_pos_interacted_items_in_impressions', 'pos_last_interaction_in_impressions'], axis=1, inplace=True)
        print(train_df.shape[1])

        if self.features_to_one_hot is not None:
            for f in self.features_to_one_hot:
                one_hot = pd.get_dummies(train_df[f])
                train_df = train_df.drop([f], axis=1)
                train_df = train_df.join(one_hot)

        # Creating univoque id for each user_id / session_id pair
        train_df = train_df.sort_values(by=['user_id', 'session_id'])
        train_df = train_df.assign(id=(train_df['user_id'] + '_' + train_df['session_id']).astype('category').cat.codes)

        train_features = train_df.drop(['user_id', 'session_id', 'label', 'id'], axis=1)

        X_train = train_features.values
        y_train = train_df['label'].values
        queries_train = train_df['id'].values

        if self.algo == 'catboost':
            features = list(train_features.columns.values)
            self.categorical_features = []
            for f in features:
                if isinstance(train_features.head(1)[f].values[0], str):
                    self.categorical_features.append(features.index(f))
                    print(f + ' is categorical!')

            if len(self.categorical_features) == 0:
                self.categorical_features = None

        # Creating pool for training data
        train_with_weights = Pool(
            data=X_train,
            label=y_train,
            group_id=queries_train,
            cat_features=self.categorical_features
        )

        test_with_weights = None

        if self.include_test:
            test_df = data.classification_test_df(
                mode=self.mode, sparse=False, cluster=self.cluster)

            test_df = test_df.sort_values(by=['user_id', 'session_id'])

            test_df['id'] = test_df.groupby(['user_id', 'session_id']).ngroup()

            X_test = test_df.drop(['user_id', 'session_id', 'label', 'id'], axis=1).values
            y_test = test_df['label'].values
            queries_test = test_df['id'].values

            print("pooling")
            test_with_weights = Pool(
                data=X_test,
                label=y_test,
                group_id=queries_test,
                cat_features=self.categorical_features
            )

        print('data for train ready')
        self.ctb = self.fit_model(self.default_parameters,
                                  train_pool=train_with_weights,
                                  test_pool=test_with_weights)
        print('fit done')

        # ----To store model----
        if self.file_to_store is not None:
            pickle.dump(self.ctb, open(self.file_to_store, 'wb'))  # pickling

    def get_scores_batch(self):
        if self.scores_batch is None:
            self.recommend_batch()
        return self.scores_batch

    def func(self, x):
        """
        Func given to progress_apply to create recommendations given a dataset for catboost
        :param x: groupd df containing same trg_idx (which is the index to return in the tuples)
        :return: tuple (trg_idx, list of recs)
        """

        target_idx = x.trg_idx.values[0]

        x = x.sort_values(by=['impression_position'])

        X_test = x.drop(['label', 'trg_idx'], axis=1).values

        # useless
        # y_test = x['label'].values
        group_id = x.trg_idx.values

        test_with_weights = Pool(
            data=X_test,
            label=None,
            group_id=group_id,
            cat_features=self.categorical_features
        )

        if self.limit_trees and self.limit_trees>0:
            scores = self.ctb.predict(test_with_weights, ntree_end=self.limit_trees)
        else:
            scores = self.ctb.predict(test_with_weights)

        impr = list(map(int, self.test_df.at[target_idx, 'impressions'].split('|')))

        min_len = len(scores)
        if len(scores) != len(impr):
            print("At session" + self.test_df.at[target_idx, 'session_id'] + 'found different len of scores wrt len '
                                                                             'of impressions')
            print(x.impression_position)
            print(impr)
            print(scores)
        if len(scores) > len(impr):
            min_len = len(impr)

        scores_impr = [[scores[i], impr[i]] for i in range(min_len)]

        # Order by max score
        scores_impr.sort(key=lambda x: x[0], reverse=True)

        preds = [x[1] for x in scores_impr]
        scores = [x[0] for x in scores_impr]

        self.predictions.append((target_idx, preds))
        self.scores_batch.append((target_idx, preds, scores))

    def recommend_batch(self):

        test_df = data.classification_test_df(
            mode=self.mode, sparse=False, cluster=self.cluster, algo=self.algo).copy()

        test_df = test_df.sort_values(by=['user_id', 'session_id', 'impression_position'])

        #test_df.drop(['avg_price_interacted_item','average_price_position', 'avg_pos_interacted_items_in_impressions', 'pos_last_interaction_in_impressions'], axis=1, inplace=True)
        if len(self.features_to_drop) > 0:
            test_df = test_df.reindex(
                'user_id,session_id,label,timing_last_action_before_clk,search_for_poi_distance_from_last_clickout,search_for_poi_distance_from_first_action,change_sort_order_distance_from_last_clickout,change_sort_order_distance_from_first_action,impression_position_wrt_last_interaction,impression_position_wrt_second_last_interaction,times_impression_appeared,time_per_step,time_elapsed_from_last_time_impression_appeared,tablet,steps_from_last_time_impression_appeared,sorted by default,sort by rating only,sort by rating and recommended,sort by price only,sort by price and recommended,sort by our recommendations,sort by interaction sort button,sort by distance only,sort by distance and recommended,session_length_in_time,session_length_in_step,search_for_item_session_ref_this_impr,search_for_item_session_ref_not_in_impr,price_position,price,pos_last_interaction_in_impressions,popularity,mobile,last_time_reference_did_not_appeared,last_time_impression_appeared_as_search_for_item,last_time_impression_appeared_as_interaction_item_rating,last_time_impression_appeared_as_interaction_item_info,last_time_impression_appeared_as_interaction_item_image,last_time_impression_appeared_as_interaction_item_deals,last_time_impression_appeared_as_clickout_item,interaction_item_rating_session_ref_this_impr,interaction_item_rating_session_ref_not_in_impr,interaction_item_info_session_ref_this_impr,interaction_item_info_session_ref_not_in_impr,interaction_item_image_session_ref_this_impr,interaction_item_image_session_ref_not_in_impr,interaction_item_deals_session_ref_this_impr,interaction_item_deals_session_ref_not_in_impr,impression_position,frenzy_factor,desktop,clickout_item_session_ref_this_impr,clickout_item_session_ref_not_in_impr,avg_price_interacted_item,avg_pos_interacted_items_in_impressions,average_price_position,accomodation feature WiFi (Rooms),accomodation feature WiFi (Public Areas),accomodation feature Wheelchair Accessible,accomodation feature Water Slide,accomodation feature Washing Machine,accomodation feature Volleyball,accomodation feature Very Good Rating,accomodation feature Towels,accomodation feature Theme Hotel,accomodation feature Terrace (Hotel),accomodation feature Tennis Court (Indoor),accomodation feature Tennis Court,accomodation feature Television,accomodation feature Teleprinter,accomodation feature Telephone,accomodation feature Table Tennis,accomodation feature Szep Kartya,accomodation feature Swimming Pool (Outdoor),accomodation feature Swimming Pool (Indoor),accomodation feature Swimming Pool (Combined Filter),accomodation feature Swimming Pool (Bar),accomodation feature Surfing,accomodation feature Sun Umbrellas,accomodation feature Steam Room,accomodation feature Spa Hotel,accomodation feature Spa (Wellness Facility),accomodation feature Solarium,accomodation feature Skiing,accomodation feature Ski Resort,accomodation feature Sitting Area (Rooms),accomodation feature Singles,accomodation feature Shower,accomodation feature Shooting Sports,accomodation feature Serviced Apartment,accomodation feature Senior Travellers,accomodation feature Self Catering,accomodation feature Sauna,accomodation feature Satisfactory Rating,accomodation feature Satellite TV,accomodation feature Sailing,accomodation feature Safe (Rooms),accomodation feature Safe (Hotel),accomodation feature Room Service (24/7),accomodation feature Room Service,accomodation feature Romantic,accomodation feature Restaurant,accomodation feature Resort,accomodation feature Reception (24/7),accomodation feature Radio,accomodation feature Pousada (BR),accomodation feature Porter,accomodation feature Pool Table,accomodation feature Playground,accomodation feature Pet Friendly,accomodation feature Organised Activities,accomodation feature Openable Windows,accomodation feature On-Site Boutique Shopping,accomodation feature Non-Smoking Rooms,accomodation feature Nightclub,accomodation feature Motel,accomodation feature Minigolf,accomodation feature Microwave,accomodation feature Massage,accomodation feature Luxury Hotel,accomodation feature Lift,accomodation feature Laundry Service,accomodation feature Large Groups,accomodation feature Kosher Food,accomodation feature Kids\' Club,accomodation feature Jacuzzi (Hotel),accomodation feature Ironing Board,accomodation feature Hypoallergenic Rooms,accomodation feature Hypoallergenic Bedding,accomodation feature Hydrotherapy,accomodation feature House / Apartment,accomodation feature Hotel Bar,accomodation feature Hotel,accomodation feature Hot Stone Massage,accomodation feature Hostel,accomodation feature Hostal (ES),accomodation feature Horse Riding,accomodation feature Honeymoon,accomodation feature Hiking Trail,accomodation feature Health Retreat,accomodation feature Hammam,accomodation feature Halal Food,accomodation feature Hairdryer,accomodation feature Hairdresser,accomodation feature Gym,accomodation feature Guest House,accomodation feature Good Rating,accomodation feature Golf Course,accomodation feature Gay-friendly,accomodation feature From 4 Stars,accomodation feature From 3 Stars,accomodation feature From 2 Stars,accomodation feature Fridge,accomodation feature Free WiFi (Rooms),accomodation feature Free WiFi (Public Areas),accomodation feature Free WiFi (Combined),accomodation feature Flatscreen TV,accomodation feature Fitness,accomodation feature Farmstay,accomodation feature Fan,accomodation feature Family Friendly,accomodation feature Express Check-In / Check-Out,accomodation feature Excellent Rating,accomodation feature Electric Kettle,accomodation feature Eco-Friendly hotel,accomodation feature Doctor On-Site,accomodation feature Diving,accomodation feature Direct beach access,accomodation feature Desk,accomodation feature Design Hotel,accomodation feature Deck Chairs,accomodation feature Country Hotel,accomodation feature Cot,accomodation feature Cosmetic Mirror,accomodation feature Convention Hotel,accomodation feature Convenience Store,accomodation feature Conference Rooms,accomodation feature Concierge,accomodation feature Computer with Internet,accomodation feature Club Hotel,accomodation feature Childcare,accomodation feature Central Heating,accomodation feature Casino (Hotel),accomodation feature Casa Rural (ES),accomodation feature Car Park,accomodation feature Camping Site,accomodation feature Cable TV,accomodation feature Business Hotel,accomodation feature Business Centre,accomodation feature Bungalows,accomodation feature Bowling,accomodation feature Boutique Hotel,accomodation feature Body Treatments,accomodation feature Boat Rental,accomodation feature Bike Rental,accomodation feature Bed & Breakfast,accomodation feature Beauty Salon,accomodation feature Beach Bar,accomodation feature Beach,accomodation feature Bathtub,accomodation feature Balcony,accomodation feature All Inclusive (Upon Inquiry),accomodation feature Airport Shuttle,accomodation feature Airport Hotel,accomodation feature Air Conditioning,accomodation feature Adults Only,accomodation feature Accessible Parking,accomodation feature Accessible Hotel,accomodation feature 5 Star,accomodation feature 4 Star,accomodation feature 3 Star,accomodation feature 2 Star,accomodation feature 1 Star,Wheelchair Accessible filter active when clickout,Water Slide filter active when clickout,Washing Machine filter active when clickout,Very Good Rating filter active when clickout,Top Deals filter active when clickout,Today filter active when clickout,Terrace (Hotel) filter active when clickout,Tennis Court filter active when clickout,Television filter active when clickout,Teleprinter filter active when clickout,Telephone filter active when clickout,Swimming Pool (Outdoor) filter active when clickout,Swimming Pool (Indoor) filter active when clickout,Swimming Pool (Combined Filter) filter active when clickout,Surfing filter active when clickout,Sun Umbrellas filter active when clickout,Steam Room filter active when clickout,Spa Hotel filter active when clickout,Spa (Wellness Facility) filter active when clickout,Sort by Price filter active when clickout,Sort By Rating filter active when clickout,Sort By Popularity filter active when clickout,Sort By Distance filter active when clickout,Skiing filter active when clickout,Ski Resort filter active when clickout,Singles filter active when clickout,Shower filter active when clickout,Shooting Sports filter active when clickout,Serviced Apartment filter active when clickout,Senior Travellers filter active when clickout,Self Catering filter active when clickout,Sauna filter active when clickout,Satisfactory Rating filter active when clickout,Satellite TV filter active when clickout,Safe (Hotel) filter active when clickout,Room Service filter active when clickout,Room Service (24/7) filter active when clickout,Romantic filter active when clickout,Restaurant filter active when clickout,Resort filter active when clickout,Reception (24/7) filter active when clickout,Pousada (BR) filter active when clickout,Porter filter active when clickout,Playground filter active when clickout,Pet Friendly filter active when clickout,Organised Activities filter active when clickout,Openable Windows filter active when clickout,On-Site Boutique Shopping filter active when clickout,Non-Smoking Rooms filter active when clickout,Nightclub filter active when clickout,Motel filter active when clickout,Minigolf filter active when clickout,Microwave filter active when clickout,Massage filter active when clickout,Luxury Hotel filter active when clickout,Lift filter active when clickout,Last Minute filter active when clickout,Kosher Food filter active when clickout,Kitchen filter active when clickout,Kids\' Club filter active when clickout,Jacuzzi (Hotel) filter active when clickout,Internet (Rooms) filter active when clickout,Hypoallergenic Bedding filter active when clickout,House / Apartment filter active when clickout,Hotel filter active when clickout,Hotel Chain filter active when clickout,Hotel Bar filter active when clickout,Hostel filter active when clickout,Hostal (ES) filter active when clickout,Honeymoon filter active when clickout,Holiday filter active when clickout,Health Retreat filter active when clickout,Hammam filter active when clickout,Halal Food filter active when clickout,Hairdryer filter active when clickout,Gym filter active when clickout,Guest House filter active when clickout,Good Rating filter active when clickout,Golf Course filter active when clickout,Gay Friendly filter active when clickout,From 2 Stars filter active when clickout,Fridge filter active when clickout,Free WiFi (Rooms) filter active when clickout,Free WiFi (Public Areas) filter active when clickout,Free WiFi (Combined) filter active when clickout,Focus on Rating filter active when clickout,Focus on Distance filter active when clickout,Flatscreen TV filter active when clickout,Fitness filter active when clickout,Farmstay filter active when clickout,Family Friendly filter active when clickout,Express Check-In / Check-Out filter active when clickout,Excellent Rating filter active when clickout,Diving filter active when clickout,Disneyland Paris filter active when clickout,Direct beach access filter active when clickout,Deck Chairs filter active when clickout,Deals + Beach (NL/BE) filter active when clickout,Deals + Beach (MX) filter active when clickout,Country Hotel filter active when clickout,Cot filter active when clickout,Convenience Store filter active when clickout,Conference Rooms filter active when clickout,Childcare filter active when clickout,Cheap filter active when clickout,Central Heating filter active when clickout,Casino (Hotel) filter active when clickout,Casa Rural (ES) filter active when clickout,Car Park filter active when clickout,Camping Site filter active when clickout,Business Hotel filter active when clickout,Business Centre filter active when clickout,Bungalows filter active when clickout,Breakfast Included filter active when clickout,Boutique Hotel filter active when clickout,Best Value filter active when clickout,Best Rates filter active when clickout,Bed & Breakfast filter active when clickout,Beach filter active when clickout,Beach Bar filter active when clickout,Bathtub filter active when clickout,Balcony filter active when clickout,All Inclusive (Upon Inquiry) filter active when clickout,Airport Shuttle filter active when clickout,Airport Hotel filter active when clickout,Air Conditioning filter active when clickout,Adults Only filter active when clickout,Accessible Parking filter active when clickout,Accessible Hotel filter active when clickout,5 Star filter active when clickout,4 Star filter active when clickout,3 Star filter active when clickout,2 Star filter active when clickout,2 Nights filter active when clickout,1 Star filter active when clickout'.split(
                    ',')
                , axis=1)
            #test_df.drop(self.features_to_drop, axis=1, inplace=True)

        if 'Unnamed: 0' in test_df.columns.values:
            test_df = test_df.drop(['Unnamed: 0'], axis=1)

        test_df = test_df.drop(['times_doubleclickout_on_item'], axis=1)
        print(test_df.shape[0])
        print(test_df.shape[1])

        target_indices = data.target_indices(mode=self.mode, cluster=self.cluster)

        self.test_df = data.test_df(self.mode, self.cluster)

        sessi_target = self.test_df.loc[target_indices].session_id.values
        self.dict_session_trg_idx = dict(zip(sessi_target, target_indices))

        test_df['trg_idx'] = test_df.apply(lambda row: self.dict_session_trg_idx.get(row.session_id), axis=1)

        test_df.drop(['user_id', 'session_id'], inplace=True, axis=1)

        self.predictions = []
        self.scores_batch = []

        if self.features_to_one_hot is not None:
            for f in self.features_to_one_hot:
                one_hot = pd.get_dummies(test_df[f])
                test_df = test_df.drop([f], axis=1)
                test_df = test_df.join(one_hot)

        # while True:
        #     timeNum = input("How many iterations?")
        #     try:
        #         self.set_limit_trees(int(timeNum))
        #         break
        #     except ValueError:
        #         pass

        test_df.groupby('trg_idx', as_index=False).progress_apply(self.func)

        return self.predictions

    def set_limit_trees(self, n):
        if n > 0:
            self.limit_trees = n

if __name__ == '__main__':
    model = CatboostRanker(mode='small', cluster='no_cluster', iterations=10, include_test=False, algo='xgboost')
    model.evaluate(send_MRR_on_telegram=False)
