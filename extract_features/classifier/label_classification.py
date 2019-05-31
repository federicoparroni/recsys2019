from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

def _reinsert_clickout(df):
    # take the row of the missing clickout
    clickout_rows_df = df[(df['action_type'] == 'clickout item') & df['reference'].isnull()]
    # check if it exsists
    if len(clickout_rows_df)>0:
        # retrieve from the full_df the clickout
        missing_click = data.full_df().loc[clickout_rows_df.index[0]]['reference']
        # reinsert the clickout on the df
        df.at[clickout_rows_df.index[0], 'reference']= missing_click
    return df

class LabelClassification(FeatureBase):

    """
    say for each impression of a clickout if it is the one clicked (1) or no 0
    | user_id | session_id | item_id | label
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'label_classification'
        super(LabelClassification, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        df = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        if self.mode in ['small', 'local']:
            print('reinserting clickout')
            test = test.groupby(['session_id', 'user_id']).progress_apply(_reinsert_clickout)
            df = pd.concat([df, test])
        df = df[(df.action_type == "clickout item") & (df.reference.notnull())]
        df = df.drop_duplicates("session_id", keep="last")
        labels = list()
        for index, row in tqdm(df.iterrows(), desc="Scanning df to generate labels"):
            reference = int(row.reference)
            impressions = list(map(int, row.impressions.split("|")))
            if reference in impressions and impressions.index(reference) == 0:
                labels.append(1)
            else:
                labels.append(0)

        df = df[["user_id", "session_id"]]

        #add label for prediction on full_df
        if self.mode == "full":
            print("Adding full test rows")
            test = test[test.action_type == "clickout item"].drop_duplicates("session_id", keep="last")
            test = test[["user_id", "session_id"]]
            df = pd.concat([df, test])
        df["label"] = labels
        return df

if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = LabelClassification(mode=mode, cluster='no_cluster')
    c.save_feature()
