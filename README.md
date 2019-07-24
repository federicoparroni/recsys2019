# ACM RecSys Challenge 2019
<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="Recommender System 2018 Challenge Polimi" />
</p>
<p align="center">
    <img src="https://recsys.trivago.cloud/site/templates/images/logo.svg" />
</p>
<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" />
</p>

## About the challenge
[Trivago](https://www.trivago.com) is a global hotel search platform focused on reshaping the way travelers search for and compare hotels, while enabling advertisers of hotels to grow their businesses by providing access to a broad audience of travelers via our websites and apps. We provide aggregated information about the characteristics of each accommodation to help travelers to make an informed decision and find their ideal place to stay. Once a choice is made, the users get redirected to the selected booking site to complete the booking.

It’s in the interest of the traveler, advertising booking site, and trivago to suggest suitable accommodations that fit the needs of the traveler best to increase the chance of a redirect (click­out) to a booking site. We face a few challenges when it comes to recommending the best options for our visitors, so it’s important to effectively make use of the explicit and implicit user signals within a session (clicks, search refinement, filter usage) to detect the users’ intent as quickly as possible and to update the recommendations to tailor the result list to these needs.

**Goal of the challenge is to develop a session-based and context-aware recommender system to adapt a list of accommodations according to the needs of the user. In the challenge, participants will have to predict which accommodations have been clicked in the search result during the last part of a user session.** Afterwards predictions are evaluated offline and scores will be displayed in a leaderboard.

Visit [the challenge website](https://recsys.trivago.cloud/challenge/) for more information about the challenge.

You can read the article in which we describe our work [here](paper/article_acm2019.pdf).

## Team members
We partecipated in the challenge as PoliCloud8, a team of 8 MSc students from Politecnico di Milano:

* **[D'Amico Edoardo](https://github.com/damicoedoardo)**
* **[Gabbolini Giovanni](https://github.com/GiovanniGabbolini)**
* **[Montesi Daniele](https://github.com/danmontesi)**
* **[Moreschini Matteo](https://github.com/teomores)**
* **[Parroni Federico](https://github.com/keyblade95)**
* **[Piccinini Federico](https://github.com/APPiccio)**
* **[Rossettini Alberto](https://github.com/albeRoss)**
* **[Russo Introito Alessio](https://github.com/russointroitoa)**

We worked under the supervision of two PhD students:
* **[Cesare Bernardis](https://github.com/cesarebernardis)**
* **[Maurizio Ferrari Dacrema](https://github.com/maurizioFD)**

## Dataset
The dataset can be found at this link: [challenge dataset page](https://recsys.trivago.cloud/challenge/dataset). You must be registered to download it.
<br/>
<br/>

## Run the code
### Preprocessing
Run this to preprocess the original data:
````python
python preprocess.py
````
Files will be saved inside: dataset/preprocessed. It is possibile to work with different modes:
* *full*: all the samples from train.csv are used as training set and all the test samples are used as test set
* *local*: 80% of the train.csv is used, the remaining 20% is stashed as validation set
* *small*: only a small number of sample is taken from the train.csv (you can choose how many, default is 100k samples). This is useful for debugging purposes.

### Training and test set
Inside the folder `preprocess_utils`, there are files used to create the datasets suitable for each model:

**Core models**
* `dataset_xgboost.py`: create the dataset for XGBoost
* `create_dataset_classification_rnn.py`: create the dataset for RNN
* `tfranking_dataset_creator_2.py`: create the dataset for TensorflowRanking
* `dataset_catboost.py.py`: create the dataset for Catboost
* `dataset_lightGBM.py.py`: create the dataset for LightGBM

**Support models**
* `dataset_xgboost_classifier.py`: create the dataset for XGBoost classifier
* `create_dataset_binary_classification_rnn.py`: create the dataset for RNN classifier

**Stacking Ensemble**
* `dataset_stacking.py`: create the dataset for the Stacking Ensemble

### Features
One of the hardest phase of the competition was *feature engineering*. We managed to craft around a hundred different features. You can find all of them in the folder `extract_features`. Each file creates a single feature and contains a description of it.

### Models
You can find all the recommenders we developed inside the folder `recommenders`.
They inherit from a common base class called `RecommenderBase`. This abstract class exposes some useful methods, like:

```python
fit()
```
Fit the model on the data. Inherited class should extend this method in the appropriate way.
<br/>
<br/>
```python
recommend_batch()
```
Return the list of recommendations.
<br/>
<br/>
```python
evaluate()
```
Return the MRR computed on the local test set.
<br/>
<br/>

We developed several models that have been finally ensembled:
   * XGBoost
   * Tensorflow Ranking
   * Recurrent Neural Network
   * Catboost
   * LightGBM

Read the paper for more details.
<br/>
<br/>
## Ensemble
We trained a new model in a new training set composed by the predictions of the previously trained models (stacking), together with all the other feature we crafted.

We report below an analysis of the **permutation importance** of the various features of the ensemble.

<table>
<thead><tr style="display:block;"><td style="width:110px;"><b>Weight</b></td><td><b>Feature</b></td></tr></thead>
<tbody style="display:block; overflow:auto; height:400px;">
<tr><td>0.4177 ± 0.0013</td><td>xgb_700</td></tr>
<tr><td>0.0029 ± 0.0003</td><td>scores_softmax_loss</td></tr>
<tr><td>0.0027 ± 0.0002</td><td>rnn_no_bias_balanced</td></tr>
<tr><td>0.0025 ± 0.0003</td><td>rnn_classifier</td></tr>
<tr><td>0.0009 ± 0.0003</td><td>xgboost_impr_features</td></tr>
<tr><td>0.0009 ± 0.0002</td><td>past_actions_involving_impression_session_clickout_item</td></tr>
<tr><td>0.0008 ± 0.0003</td><td>personalized_popularity</td></tr>
<tr><td>0.0007 ± 0.0001</td><td>future_actions_involving_impression_session_clickout_item</td></tr>
<tr><td>0.0005 ± 0.0002</td><td>num_impressoins_in_clickout</td></tr>
<tr><td>0.0004 ± 0.0002</td><td>last_pos_interacted</td></tr>
<tr><td>0.0003 ± 0.0001</td><td>scores_pairwise_soft_zero_one_loss</td></tr>
<tr><td>0.0003 ± 0.0000</td><td>num_clickouts</td></tr>
<tr><td>0.0003 ± 0.0003</td><td>length_timestamp</td></tr>
<tr><td>0.0003 ± 0.0002</td><td>past_time_from_closest_interaction_impression</td></tr>
<tr><td>0.0002 ± 0.0002</td><td>rnn_GRU_2layers_64units_2dense_noclass0</td></tr>
<tr><td>0.0002 ± 0.0001</td><td>top_pop_per_impression</td></tr>
<tr><td>0.0002 ± 0.0001</td><td>impression_position</td></tr>
<tr><td>0.0002 ± 0.0002</td><td>elapsed_last_action_click</td></tr>
<tr><td>0.0002 ± 0.0002</td><td>session_length_timestamp</td></tr>
<tr><td>0.0002 ± 0.0001</td><td>step_from_last_interaction</td></tr>
<tr><td>0.0002 ± 0.0001</td><td>future_time_from_closest_interaction_impression</td></tr>
<tr><td>0.0002 ± 0.0001</td><td>impression_time</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>fraction_pos_price</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>times_impression_appeared_in_clickouts_session</td></tr>
<tr><td>0.0001 ± 0.0002</td><td>frenzy_factor</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>platform_similarity</td></tr>
<tr><td>0.0001 ± 0.0000</td><td>past_actions_involving_impression_session_search_for_item</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>timestamp_from_last_interaction</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>feature</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>mean_price_interacted</td></tr>
<tr><td>0.0001 ± 0.0002</td><td>mean_pos_interacted</td></tr>
<tr><td>0.0001 ± 0.0000</td><td>actions_num_ref_diff_from_impressions_clickout_item</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>mean_time_action</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>past_mean_cheap_pos_interacted</td></tr>
<tr><td>0.0001 ± 0.0002</td><td>variance_last_action</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>price</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>mean_time_per_step</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>scores_cosine</td></tr>
<tr><td>0.0001 ± 0.0001</td><td>destination_change_distance_from_first_action</td></tr>
<tr><td>0 ± 0.0001     </td><td>max_pos_interacted</td></tr>
<tr><td>0 ± 0.0001     </td><td>percentage_of_total_city_clk</td></tr>
<tr><td>0 ± 0.0001     </td><td>perc_click_appeared</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_pos_clicked_4_8</td></tr>
<tr><td>0 ± 0.0002     </td><td>elapsed_last_action_click_log</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_pos_clicked_4_8</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_pos_clicked_2</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_pos_clicked_3</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_involving_impression_session_clickout_item</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_pos_closest_reference</td></tr>
<tr><td>0 ± 0.0001     </td><td>session_length_step</td></tr>
<tr><td>0 ± 0.0001     </td><td>length_step</td></tr>
<tr><td>0 ± 0.0000     </td><td>num_interactions_impr</td></tr>
<tr><td>0 ± 0.0000     </td><td>search_for_poi_distance_from_first_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>search_for_poi_distance_from_last_clickout</td></tr>
<tr><td>0 ± 0.0000     </td><td>satisfaction_percentage</td></tr>
<tr><td>0 ± 0.0001     </td><td>top_pop_interaction_clickout_per_impression</td></tr>
<tr><td>0 ± 0.0001     </td><td>past_actions_involving_impression_session_item_deals</td></tr>
<tr><td>0 ± 0.0001     </td><td>past_times_interacted_impr</td></tr>
<tr><td>0 ± 0.0001     </td><td>num_times_item_impressed</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_num_ref_diff_from_impressions_interaction_item_rating</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_num_ref_diff_from_impressions_interaction_item_info</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_pos_clicked_9_15</td></tr>
<tr><td>0 ± 0.0001     </td><td>perc_clickouts</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_session_num</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_involving_impression_session_interaction_item_rating</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_num_ref_diff_from_impressions_search_for_item</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_pos_clicked_16_25</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_pos_clicked_3</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_involving_impression_session_interaction_item_info</td></tr>
<tr><td>0 ± 0.0001     </td><td>destination_change_distance_from_last_clickout</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_FI</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_search for item</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_FR</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_CZ</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_search for destination</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_interaction item rating</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_search for poi</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_GR</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_DE</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_no_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_CH</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_AR</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_CO</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_AA</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_AE</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_AT</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_AU</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_BE</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_BG</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_ES</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_interaction item info</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_EC</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_BR</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_CN</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_DK</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_CL</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_CA</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_closest_action_involving_impression_no_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>day_6</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_interaction item image</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_closest_action_involving_impression_no_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>price_log</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_involving_impression_clickout item</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_involving_impression_interaction item deals</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_involving_impression_interaction item image</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_involving_impression_interaction item info</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_involving_impression_interaction item rating</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_involving_impression_no_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_involving_impression_search for item</td></tr>
<tr><td>0 ± 0.0000     </td><td>session_device_d</td></tr>
<tr><td>0 ± 0.0000     </td><td>session_device_m</td></tr>
<tr><td>0 ± 0.0000     </td><td>session_device_t</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_closest_action_involving_impression_search for item</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_closest_action_involving_impression_not_present</td></tr>
<tr><td>0 ± 0.0000     </td><td>std_last_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>day_0</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_interaction item deals</td></tr>
<tr><td>0 ± 0.0000     </td><td>day_1</td></tr>
<tr><td>0 ± 0.0000     </td><td>day_2</td></tr>
<tr><td>0 ± 0.0000     </td><td>day_3</td></tr>
<tr><td>0 ± 0.0000     </td><td>day_4</td></tr>
<tr><td>0 ± 0.0000     </td><td>day_5</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_HR</td></tr>
<tr><td>0 ± 0.0000     </td><td>moment_A</td></tr>
<tr><td>0 ± 0.0000     </td><td>moment_E</td></tr>
<tr><td>0 ± 0.0000     </td><td>moment_M</td></tr>
<tr><td>0 ± 0.0000     </td><td>moment_N</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_closest_action_involving_impression_interaction item rating</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_change of sort order</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_clickout item</td></tr>
<tr><td>0 ± 0.0000     </td><td>last_action_filter selection</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_HK</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_KR</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_HU</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_TR</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_UK</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_US</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_UY</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_VN</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_ZA</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_closest_action_involving_impression_interaction item  |image
<tr><td>0 ± 0.0000     </td><td>sort_order_active_when_clickout_distance and recommended</td></tr>
<tr><td>0 ± 0.0000     </td><td>sort_order_active_when_clickout_distance only</td></tr>
<tr><td>0 ± 0.0000     </td><td>sort_order_active_when_clickout_interaction sort button</td></tr>
<tr><td>0 ± 0.0000     </td><td>sort_order_active_when_clickout_our recommendations</td></tr>
<tr><td>0 ± 0.0000     </td><td>sort_order_active_when_clickout_price and recommended</td></tr>
<tr><td>0 ± 0.0000     </td><td>sort_order_active_when_clickout_price only</td></tr>
<tr><td>0 ± 0.0000     </td><td>sort_order_active_when_clickout_rating and recommended</td></tr>
<tr><td>0 ± 0.0000     </td><td>sort_order_active_when_clickout_rating only</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_num_ref_diff_from_impressions_no_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_ID</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_involving_impression_session_no_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_closest_action_involving_impression_interaction item  |deals
<tr><td>0 ± 0.0000     </td><td>future_closest_action_involving_impression_clickout item</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_closest_action_involving_impression_search for item</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_closest_action_involving_impression_not_present</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_closest_action_involving_impression_clickout item</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_closest_action_involving_impression_interaction item deals</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_closest_action_involving_impression_interaction item image</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_closest_action_involving_impression_interaction item info</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_TW</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_involving_impression_session_interaction_item_deals</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_TH</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_PH</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_IE</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_IL</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_IN</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_IT</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_JP</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_closest_action_involving_impression_interaction item rating</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_MX</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_MY</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_NL</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_NO</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_NZ</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_PE</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_PL</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_closest_action_involving_impression_interaction item info</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_RU</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_SI</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_SG</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_SE</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_PT</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_RS</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_RO</td></tr>
<tr><td>0 ± 0.0000     </td><td>platform_SK</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_actions_involving_impression_session_item_info</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_actions_involving_impression_session_item_rating</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_mean_pos</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_involving_impression_session_search_for_item</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_num_ref_diff_from_impressions_interaction_item_deals</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_position_impression_same_closest_clickout</td></tr>
<tr><td>0 ± 0.0001     </td><td>mean_price_clickout</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_times_impr_appeared</td></tr>
<tr><td>0 ± 0.0000     </td><td>change_sort_order_distance_from_last_clickout</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_pos_clicked_2</td></tr>
<tr><td>0 ± 0.0001     </td><td>mean_cheap_pos_interacted</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_actions_involving_impression_session_item_rating</td></tr>
<tr><td>0 ± 0.0001     </td><td>percentage_of_total_plat_inter</td></tr>
<tr><td>0 ± 0.0001     </td><td>future_actions_involving_impression_session_search_for_item</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_mean_cheap_pos_interacted</td></tr>
<tr><td>0 ± 0.0000     </td><td>actions_involving_impression_session_interaction_item_image</td></tr>
<tr><td>0 ± 0.0001     </td><td>future_pos_closest_reference</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_mean_price_interacted</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_pos_clicked_1</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_mean_price_interacted</td></tr>
<tr><td>0 ± 0.0002     </td><td>percentage_of_total_city_inter</td></tr>
<tr><td>0 ± 0.0001     </td><td>past_times_user_interacted_impression</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_session_num</td></tr>
<tr><td>0 ± 0.0001     </td><td>actions_num_ref_diff_from_impressions_interaction_item_image</td></tr>
<tr><td>0 ± 0.0001     </td><td>scores_manhatthan</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_pos_clicked_1</td></tr>
<tr><td>0 ± 0.0001     </td><td>change_sort_order_distance_from_first_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_mean_pos_impr_appeared</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_position_impression_same_closest_clickout</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_actions_involving_impression_session_item_image</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_mean_pos</td></tr>
<tr><td>0 ± 0.0001     </td><td>impression_pos_price</td></tr>
<tr><td>0 ± 0.0001     </td><td>past_actions_involving_impression_session_no_action</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_times_interacted_impr</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_pos_clicked_16_25</td></tr>
<tr><td>0 ± 0.0001     </td><td>rating</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_mean_pos_impr_appeared</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_times_impr_appeared</td></tr>
<tr><td>0 ± 0.0000     </td><td>future_actions_involving_impression_session_item_info</td></tr>
<tr><td>0 ± 0.0000     </td><td>past_actions_involving_impression_session_item_image</td></tr>
<tr><td>-0.0001 ± 0.0000</td><td>future_actions_involving_impression_session_no_action</td></tr>
<tr><td>-0.0001 ± 0.0001</td><td>future_actions_involving_impression_session_item_deals</td></tr>
<tr><td>-0.0001 ± 0.0000</td><td>future_times_user_interacted_impression</td></tr>
<tr><td>-0.0001 ± 0.0000</td><td>future_pos_clicked_9_15</td></tr>
<tr><td>-0.0001 ± 0.0001</td><td>percentage_of_total_plat_clk</td></tr>
<tr><td>-0.0001 ± 0.0001</td><td>stars</td></tr>
</tbody>
</table>
