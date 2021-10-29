#please refer to README.md for fuller explanation
from typing import ClassVar
import fire
import numpy as np
import pandas as pd
import itertools #to list the films
import sys
import os
import pickle #to save the model
from ast import literal_eval #to extract text from dictionaries
sys.path.append("../../")

#hybrid model of choice
import lightfm
from lightfm import LightFM, cross_validation
from lightfm.data import Dataset

#subdirectory from Microsoft GitHub devoted to recommenders - has useful metrics used here
from reco_utils.evaluation.python_evaluation import precision_at_k, recall_at_k

#I/O files
src1 = os.path.join(r'data/', 'ratings_small.csv')
src2 = os.path.join(r'data/', 'movies_metadata_extract.csv')
op = os.path.join(r'data/', 'the_prediction_file.csv')
md = os.path.join(r'model/', 'the_model_file.sav')

#udfs
def load_data(infile1, infile2):
    '''
    *DESCRIPTION: to bring in ratings + extracted metadata tables
    *preprocess by extracting dictionary values
    *then join, rename + drop as required
    *OUTPUT: clean base dataset
    '''
    a = pd.read_csv(infile1)
    a.rename(columns={'userId': 'userID','movieId':'itemID'}, inplace=True)

    #chose genre as the text feature (item attribute)
    df = pd.read_csv(infile2)[['id','genres']]
    df2 = df.drop(df[df['genres'] == '[]'].index, axis=0).reset_index(drop=True)
    df2['genres'] = df2['genres'].apply(lambda x: [x['name'] for x in literal_eval(x)])
    df2.rename(columns={'id': 'itemID'}, inplace=True)

    movie_genre = df2['genres']
    all_movie_genre = sorted(list(set(itertools.chain.from_iterable(movie_genre))))

    #make a single base dataset
    new_data = a.join(df2, on='itemID', how='left', lsuffix='a')
    new_data.drop(['timestamp','itemID'],axis=1,inplace=True)
    new_data.rename(columns={'itemIDa': 'itemID'}, inplace=True)

    dataset = Dataset()
    dataset.fit(a['userID'], a['itemID'], item_features=all_movie_genre)
    item_features = dataset.build_item_features((x, y) for x,y in zip(a.itemID, movie_genre))

    (interactions, weights) = dataset.build_interactions(a.iloc[:, 0:3].values)

    return a, movie_genre, item_features, interactions, weights, all_movie_genre
    

def split_data(ints):
    '''
    DESCRIPTION: to randomly assign 3:1 split for train/holdout (seeded)
    OUTPUT: partitioned datasets for modelling/prediction
    '''
    train_interactions, test_interactions = cross_validation.random_train_test_split(ints, test_percentage=0.25, random_state=np.random.RandomState(42))
    
    return train_interactions, test_interactions


def model_data(trnint, atts, rnk, alpha):
    '''
    DESCRIPTION: fit lightfm with tuned hyperparams (seeded) + genre text attribute.  Hyperparams are depth rank + increment degree.
    OUTPUT: fit model with params
    '''
    model = LightFM(loss='warp', no_components=rnk, learning_rate=alpha, 
                     item_alpha=1e-6, user_alpha=1e-6, random_state=np.random.RandomState(42))

    model.fit(interactions=trnint, item_features=atts, epochs=20)
    
    return model


def pre_prep(ints,inp,lst1,lst2):
    '''
    DESCRIPTION: some staging params for producing actual test data
    '''
    uids, iids, interaction_data = cross_validation._shuffle(ints.row, ints.col
                            , ints.data, random_state=np.random.RandomState(42))
    
    cutoff = int((1.0 - 0.25) * len(uids))
    
    test_idx = slice(cutoff, None)
    
    dataset = Dataset()
    dataset.fit(a['userID'], a['itemID'], item_features=all_movie_genre)
    item_features = dataset.build_item_features((x, y) for x,y in zip(a.itemID, movie_genre))
    (interactions, weights) = dataset.build_interactions(a.iloc[:, 0:3].values)
    
    uid_map, ufeature_map, iid_map, ifeature_map = dataset.mapping()
    
    return test_idx, uids, iids, uid_map, iid_map


def prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights):
    '''
    DESCRIPTION: take above params and get test dataset for evaluation
    OUTPUT: actual ratings for test data, mapped on id
    '''
    test_df = pd.DataFrame(
        zip(
            uids[test_idx],
            iids[test_idx],
            [list(uid_map.keys())[x] for x in uids[test_idx]],
            [list(iid_map.keys())[x] for x in iids[test_idx]],
        ),
        columns=["uid", "iid", "userID", "itemID"],
    )

    dok_weights = weights.todok()
    test_df["rating"] = test_df.apply(lambda x: dok_weights[x.uid, x.iid], axis=1)

    return test_df[["userID", "itemID", "rating"]]


def get_predictions(data, uid_map, iid_map, interactions, model, num_threads
                   , user_features, item_features):
    '''
    DESCRIPTION: get predicted ratings for test set - note the longest part of code (approx 20-40 mins)
    OUTPUT: predicted ratings for test data, mapped on id
    '''
    #num_threads: note my PC only has 2 cores - suggest to set this to cores of your PC
    
    users, items = [], []
    
    item = list(data.itemID.unique())
    
    for user in data.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
        
    all_predictions = pd.DataFrame(data={"userID": users, "itemID": items})
    all_predictions["uid"] = all_predictions.userID.map(uid_map)
    all_predictions["iid"] = all_predictions.itemID.map(iid_map)
    dok_weights = interactions.todok()
    all_predictions["rating"] = all_predictions.apply(lambda x: dok_weights[x.uid, x.iid], axis=1)
    all_predictions = all_predictions[all_predictions.rating < 1].reset_index(drop=True)
    all_predictions = all_predictions.drop("rating", axis=1)
    
    all_predictions['prediction'] = all_predictions.apply(lambda x: model.predict(user_ids=int(x["uid"]),
                                              item_ids=[x["iid"]],
            user_features=user_features,item_features=item_features,num_threads=num_threads,)[0], axis=1,)
    
    return all_predictions[['userID','itemID','prediction']]


def eval_metrics(rating_true, rating_pred, k):
    '''
    DESCRIPTION: calc some key metrics to evaluate quality of recs
    OUTPUT: precision + recall, both between 0 and 1
    '''
    precision = precision_at_k(rating_true=rating_true, rating_pred=rating_pred, k=k)

    recall = recall_at_k(rating_true=rating_true, rating_pred=rating_pred, k=k)

    print(f"Precision@K:\t{precision:.3f}", f"Recall@K:\t{recall:.3f}", sep='\n')
    

def save_file(dset, to):
    '''
    DESCRIPTION: save the predicted ratings
    OUTPUT: .csv file of predicted ratings for all movies per user (higher the better) 
    '''
    return dset.to_csv(to ,index=False)


def save_model(mdl, to):
    '''
    DESCRIPTION: save the best model with params to model folder
    OUTPUT: .sav file
    '''
    return pickle.dump(mdl, open(to, 'wb'))


#execute the above
if __name__ == '__main__':
  a, movie_genre, item_features, interactions, weights, all_movie_genre = load_data(infile1=src1,infile2=src2)
  
  train_interactions, test_interactions = split_data(interactions)
  
  (model = model_data(train_interactions, item_features, rnk=25, alpha=0.25)

  test_idx, uids, iids, uid_map, iid_map = pre_prep(interactions, a, all_movie_genre, movie_genre)
  
  test_df = prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights)
  
  all_predictions = get_predictions(data=a, uid_map=uid_map, iid_map=iid_map, interactions=train_interactions
                , model=model, num_threads=2, user_features=None, item_features=item_features)
                
  eval_metrics(rating_true=test_df, rating_pred=all_predictions, k=10)
  
  save_model(model, md)
  
  save_file(all_predictions, op)

  #fire.Fire(model)

#END
