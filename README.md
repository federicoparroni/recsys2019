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
Trivago is a global hotel search platform focused on reshaping the way travelers search for and compare hotels, while enabling advertisers of hotels to grow their businesses by providing access to a broad audience of travelers via our websites and apps. We provide aggregated information about the characteristics of each accommodation to help travelers to make an informed decision and find their ideal place to stay. Once a choice is made, the users get redirected to the selected booking site to complete the booking.

It’s in the interest of the traveler, advertising booking site, and trivago to suggest suitable accommodations that fit the needs of the traveler best to increase the chance of a redirect (click­out) to a booking site. We face a few challenges when it comes to recommending the best options for our visitors, so it’s important to effectively make use of the explicit and implicit user signals within a session (clicks, search refinement, filter usage) to detect the users’ intent as quickly as possible and to update the recommendations to tailor the result list to these needs.

**Goal of the challenge is to develop a session-based and context-aware recommender system to adapt a list of accommodations according to the needs of the user. In the challenge, participants will have to predict which accommodations have been clicked in the search result during the last part of a user session.** Afterwards predictions are evaluated offline and scores will be displayed in a leaderboard.

Visit [the challenge website](https://recsys.trivago.cloud/challenge/) for more information about the challenge.

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
The dataset can be found at this link: [dataset](https://recsys.trivago.cloud/challenge/dataset). You must register to download it.

## Run the code
### Preprocessing
Run this to preprocess the original data:
````python
python preprocess.py
````
Files will be saved inside: dataset/preprocessed and dataset/matrices.

### Features
One of the hardest phase of the competition was *feature engineering*. We managed to craft around a hundred different features. You can find all of them in the folder `extract_features`. Each file contains a description of the feature that it creates.

### Models
You can find all the recommenders we developed inside the folder `recommenders`.
They inherit from a common base class called `RecommenderBase`. This abstract class exposes some useful methods, like:
```python
fit()
```
Fit the model on the data. Inherited class should extend this method in the appropriate way.

```python
recommend_batch()
```
Return the list of recommendations.

```python
evaluate()
```
Return the MRR computed on the local test set.


We developed several models that have been finally ensembled:
   * XGBoost
   * Tensorflow Ranking
   * Recurrent Neural Network
   * Catboost
   * LightGBM

Read the paper for more details.

## Ensemble
We trained a new model in a new training set composed by the predictions of the previously trained models (stacking), together with all the other feature we crafted.

We report below an analysis of the *permutation importance* of the various features of the ensemble.

| Weight                         | Feature        |
|--------------------------------|----------------|
| 0.4177 ± 0.0013 | xgb_700 |
| 0.0029 ± 0.0003 | scores_softmax_loss |
| 0.0027 ± 0.0002 | rnn_no_bias_balanced |
| 0.0025 ± 0.0003 | rnn_classifier |
| 0.0009 ± 0.0003 | xgboost_impr_features |
| 0.0009 ± 0.0002 | past_actions_involving_impression_session_clickout_item |
| 0.0008 ± 0.0003 | personalized_popularity |
| 0.0007 ± 0.0001 | future_actions_involving_impression_session_clickout_item |
| 0.0005 ± 0.0002 | num_impressoins_in_clickout |
| 0.0004 ± 0.0002 | last_pos_interacted |
| 0.0003 ± 0.0001 | scores_pairwise_soft_zero_one_loss |
| 0.0003 ± 0.0000 | num_clickouts |
| 0.0003 ± 0.0003 | length_timestamp |
| 0.0003 ± 0.0002 | past_time_from_closest_interaction_impression |
| 0.0002 ± 0.0002 | rnn_GRU_2layers_64units_2dense_noclass0 |
| 0.0002 ± 0.0001 | top_pop_per_impression |
| 0.0002 ± 0.0001 | impression_position |
| 0.0002 ± 0.0002 | elapsed_last_action_click |
| 0.0002 ± 0.0002 | session_length_timestamp |
| 0.0002 ± 0.0001 | step_from_last_interaction |
| 0.0002 ± 0.0001 | future_time_from_closest_interaction_impression |
| 0.0002 ± 0.0001 | impression_time |
| 0.0001 ± 0.0001 | fraction_pos_price |
| 0.0001 ± 0.0001 | times_impression_appeared_in_clickouts_session |
| 0.0001 ± 0.0002 | frenzy_factor |
| 0.0001 ± 0.0001 | platform_similarity |
| 0.0001 ± 0.0000 | past_actions_involving_impression_session_search_for_item |
| 0.0001 ± 0.0001 | timestamp_from_last_interaction |
| 0.0001 ± 0.0001 | feature |
| 0.0001 ± 0.0001 | mean_price_interacted |
| 0.0001 ± 0.0002 | mean_pos_interacted |
| 0.0001 ± 0.0000 | actions_num_ref_diff_from_impressions_clickout_item |
| 0.0001 ± 0.0001 | mean_time_action |
| 0.0001 ± 0.0001 | past_mean_cheap_pos_interacted |
| 0.0001 ± 0.0002 | variance_last_action |
| 0.0001 ± 0.0001 | price |
| 0.0001 ± 0.0001 | mean_time_per_step |
| 0.0001 ± 0.0001 | scores_cosine |
| 0.0001 ± 0.0001 | destination_change_distance_from_first_action |
| 0 ± 0.0001      | max_pos_interacted |
| 0 ± 0.0001      | percentage_of_total_city_clk |
| 0 ± 0.0001      | perc_click_appeared |
| 0 ± 0.0000      | past_pos_clicked_4_8 |
| 0 ± 0.0002      | elapsed_last_action_click_log |
| 0 ± 0.0000      | future_pos_clicked_4_8 |
| 0 ± 0.0000      | past_pos_clicked_2 |
| 0 ± 0.0000      | past_pos_clicked_3 |
| 0 ± 0.0000      | actions_involving_impression_session_clickout_item |
| 0 ± 0.0000      | past_pos_closest_reference |
| 0 ± 0.0001      | session_length_step |
| 0 ± 0.0001      | length_step |
| 0 ± 0.0000      | num_interactions_impr |
| 0 ± 0.0000      | search_for_poi_distance_from_first_action |
| 0 ± 0.0000      | search_for_poi_distance_from_last_clickout |
| 0 ± 0.0000      | satisfaction_percentage |
| 0 ± 0.0001      | top_pop_interaction_clickout_per_impression |
| 0 ± 0.0001      | past_actions_involving_impression_session_item_deals |
| 0 ± 0.0001      | past_times_interacted_impr |
| 0 ± 0.0001      | num_times_item_impressed |
| 0 ± 0.0000      | actions_num_ref_diff_from_impressions_interaction_item_rating |
| 0 ± 0.0000      | actions_num_ref_diff_from_impressions_interaction_item_info |
| 0 ± 0.0000      | past_pos_clicked_9_15 |
| 0 ± 0.0001      | perc_clickouts |
| 0 ± 0.0000      | past_session_num |
| 0 ± 0.0000      | actions_involving_impression_session_interaction_item_rating |
| 0 ± 0.0000      | actions_num_ref_diff_from_impressions_search_for_item |
| 0 ± 0.0000      | past_pos_clicked_16_25 |
| 0 ± 0.0000      | future_pos_clicked_3 |
| 0 ± 0.0000      | actions_involving_impression_session_interaction_item_info |
| 0 ± 0.0001      | destination_change_distance_from_last_clickout |
| 0 ± 0.0000      | platform_FI |
| 0 ± 0.0000      | last_action_search for item |
| 0 ± 0.0000      | platform_FR |
| 0 ± 0.0000      | platform_CZ |
| 0 ± 0.0000      | last_action_search for destination |
| 0 ± 0.0000      | last_action_interaction item rating |
| 0 ± 0.0000      | last_action_search for poi |
| 0 ± 0.0000      | platform_GR |
| 0 ± 0.0000      | platform_DE |
| 0 ± 0.0000      | last_action_no_action |
| 0 ± 0.0000      | platform_CH |
| 0 ± 0.0000      | platform_AR |
| 0 ± 0.0000      | platform_CO |
| 0 ± 0.0000      | platform_AA |
| 0 ± 0.0000      | platform_AE |
| 0 ± 0.0000      | platform_AT |
| 0 ± 0.0000      | platform_AU |
| 0 ± 0.0000      | platform_BE |
| 0 ± 0.0000      | platform_BG |
| 0 ± 0.0000      | platform_ES |
| 0 ± 0.0000      | last_action_interaction item info |
| 0 ± 0.0000      | platform_EC |
| 0 ± 0.0000      | platform_BR |
| 0 ± 0.0000      | platform_CN |
| 0 ± 0.0000      | platform_DK |
| 0 ± 0.0000      | platform_CL |
| 0 ± 0.0000      | platform_CA |
| 0 ± 0.0000      | past_closest_action_involving_impression_no_action |
| 0 ± 0.0000      | day_6 |
| 0 ± 0.0000      | last_action_interaction item image |
| 0 ± 0.0000      | future_closest_action_involving_impression_no_action |
| 0 ± 0.0000      | price_log |
| 0 ± 0.0000      | last_action_involving_impression_clickout item |
| 0 ± 0.0000      | last_action_involving_impression_interaction item deals |
| 0 ± 0.0000      | last_action_involving_impression_interaction item image |
| 0 ± 0.0000      | last_action_involving_impression_interaction item info |
| 0 ± 0.0000      | last_action_involving_impression_interaction item rating |
| 0 ± 0.0000      | last_action_involving_impression_no_action |
| 0 ± 0.0000      | last_action_involving_impression_search for item |
| 0 ± 0.0000      | session_device_d |
| 0 ± 0.0000      | session_device_m |
| 0 ± 0.0000      | session_device_t |
| 0 ± 0.0000      | future_closest_action_involving_impression_search for item |
| 0 ± 0.0000      | future_closest_action_involving_impression_not_present |
| 0 ± 0.0000      | std_last_action |
| 0 ± 0.0000      | day_0 |
| 0 ± 0.0000      | last_action_interaction item deals |
| 0 ± 0.0000      | day_1 |
| 0 ± 0.0000      | day_2 |
| 0 ± 0.0000      | day_3 |
| 0 ± 0.0000      | day_4 |
| 0 ± 0.0000      | day_5 |
| 0 ± 0.0000      | platform_HR |
| 0 ± 0.0000      | moment_A |
| 0 ± 0.0000      | moment_E |
| 0 ± 0.0000      | moment_M |
| 0 ± 0.0000      | moment_N |
| 0 ± 0.0000      | future_closest_action_involving_impression_interaction item rating |
| 0 ± 0.0000      | last_action_change of sort order |
| 0 ± 0.0000      | last_action_clickout item |
| 0 ± 0.0000      | last_action_filter selection |
| 0 ± 0.0000      | platform_HK |
| 0 ± 0.0000      | platform_KR |
| 0 ± 0.0000      | platform_HU |
| 0 ± 0.0000      | platform_TR |
| 0 ± 0.0000      | platform_UK |
| 0 ± 0.0000      | platform_US |
| 0 ± 0.0000      | platform_UY |
| 0 ± 0.0000      | platform_VN |
| 0 ± 0.0000      | platform_ZA |
| 0 ± 0.0000      | future_closest_action_involving_impression_interaction item  |image
| 0 ± 0.0000      | sort_order_active_when_clickout_distance and recommended |
| 0 ± 0.0000      | sort_order_active_when_clickout_distance only |
| 0 ± 0.0000      | sort_order_active_when_clickout_interaction sort button |
| 0 ± 0.0000      | sort_order_active_when_clickout_our recommendations |
| 0 ± 0.0000      | sort_order_active_when_clickout_price and recommended |
| 0 ± 0.0000      | sort_order_active_when_clickout_price only |
| 0 ± 0.0000      | sort_order_active_when_clickout_rating and recommended |
| 0 ± 0.0000      | sort_order_active_when_clickout_rating only |
| 0 ± 0.0000      | actions_num_ref_diff_from_impressions_no_action |
| 0 ± 0.0000      | platform_ID |
| 0 ± 0.0000      | actions_involving_impression_session_no_action |
| 0 ± 0.0000      | future_closest_action_involving_impression_interaction item  |deals
| 0 ± 0.0000      | future_closest_action_involving_impression_clickout item |
| 0 ± 0.0000      | past_closest_action_involving_impression_search for item |
| 0 ± 0.0000      | past_closest_action_involving_impression_not_present |
| 0 ± 0.0000      | past_closest_action_involving_impression_clickout item |
| 0 ± 0.0000      | past_closest_action_involving_impression_interaction item deals |
| 0 ± 0.0000      | past_closest_action_involving_impression_interaction item image |
| 0 ± 0.0000      | past_closest_action_involving_impression_interaction item info |
| 0 ± 0.0000      | platform_TW |
| 0 ± 0.0000      | actions_involving_impression_session_interaction_item_deals |
| 0 ± 0.0000      | platform_TH |
| 0 ± 0.0000      | platform_PH |
| 0 ± 0.0000      | platform_IE |
| 0 ± 0.0000      | platform_IL |
| 0 ± 0.0000      | platform_IN |
| 0 ± 0.0000      | platform_IT |
| 0 ± 0.0000      | platform_JP |
| 0 ± 0.0000      | past_closest_action_involving_impression_interaction item rating |
| 0 ± 0.0000      | platform_MX |
| 0 ± 0.0000      | platform_MY |
| 0 ± 0.0000      | platform_NL |
| 0 ± 0.0000      | platform_NO |
| 0 ± 0.0000      | platform_NZ |
| 0 ± 0.0000      | platform_PE |
| 0 ± 0.0000      | platform_PL |
| 0 ± 0.0000      | future_closest_action_involving_impression_interaction item info |
| 0 ± 0.0000      | platform_RU |
| 0 ± 0.0000      | platform_SI |
| 0 ± 0.0000      | platform_SG |
| 0 ± 0.0000      | platform_SE |
| 0 ± 0.0000      | platform_PT |
| 0 ± 0.0000      | platform_RS |
| 0 ± 0.0000      | platform_RO |
| 0 ± 0.0000      | platform_SK |
| 0 ± 0.0000      | past_actions_involving_impression_session_item_info |
| 0 ± 0.0000      | future_actions_involving_impression_session_item_rating |
| 0 ± 0.0000      | past_mean_pos |
| 0 ± 0.0000      | actions_involving_impression_session_search_for_item |
| 0 ± 0.0000      | actions_num_ref_diff_from_impressions_interaction_item_deals |
| 0 ± 0.0000      | future_position_impression_same_closest_clickout |
| 0 ± 0.0001      | mean_price_clickout |
| 0 ± 0.0000      | past_times_impr_appeared |
| 0 ± 0.0000      | change_sort_order_distance_from_last_clickout |
| 0 ± 0.0000      | future_pos_clicked_2 |
| 0 ± 0.0001      | mean_cheap_pos_interacted |
| 0 ± 0.0000      | past_actions_involving_impression_session_item_rating |
| 0 ± 0.0001      | percentage_of_total_plat_inter |
| 0 ± 0.0001      | future_actions_involving_impression_session_search_for_item |
| 0 ± 0.0000      | future_mean_cheap_pos_interacted |
| 0 ± 0.0000      | actions_involving_impression_session_interaction_item_image |
| 0 ± 0.0001      | future_pos_closest_reference |
| 0 ± 0.0000      | past_mean_price_interacted |
| 0 ± 0.0000      | future_pos_clicked_1 |
| 0 ± 0.0000      | future_mean_price_interacted |
| 0 ± 0.0002      | percentage_of_total_city_inter |
| 0 ± 0.0001      | past_times_user_interacted_impression |
| 0 ± 0.0000      | future_session_num |
| 0 ± 0.0001      | actions_num_ref_diff_from_impressions_interaction_item_image |
| 0 ± 0.0001      | scores_manhatthan |
| 0 ± 0.0000      | past_pos_clicked_1 |
| 0 ± 0.0001      | change_sort_order_distance_from_first_action |
| 0 ± 0.0000      | future_mean_pos_impr_appeared |
| 0 ± 0.0000      | past_position_impression_same_closest_clickout |
| 0 ± 0.0000      | future_actions_involving_impression_session_item_image |
| 0 ± 0.0000      | future_mean_pos |
| 0 ± 0.0001      | impression_pos_price |
| 0 ± 0.0001      | past_actions_involving_impression_session_no_action |
| 0 ± 0.0000      | future_times_interacted_impr |
| 0 ± 0.0000      | future_pos_clicked_16_25 |
| 0 ± 0.0001      | rating |
| 0 ± 0.0000      | past_mean_pos_impr_appeared |
| 0 ± 0.0000      | future_times_impr_appeared |
| 0 ± 0.0000      | future_actions_involving_impression_session_item_info |
| 0 ± 0.0000      | past_actions_involving_impression_session_item_image |
|-0.0001 ± 0.0000 | future_actions_involving_impression_session_no_action |
|-0.0001 ± 0.0001 | future_actions_involving_impression_session_item_deals |
|-0.0001 ± 0.0000 | future_times_user_interacted_impression |
|-0.0001 ± 0.0000 | future_pos_clicked_9_15 |
|-0.0001 ± 0.0001 | percentage_of_total_plat_clk |
|-0.0001 ± 0.0001 | stars |
