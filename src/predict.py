from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import pandas as pd
import numpy as np
import os
from . import dispatcher
import joblib



TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict ():

    df = pd.read_csv (TEST_DATA)
    test_idx = df['id'].values
    predictions = None
 
    
    for FOLD in range (1):
        df = pd.read_csv (TEST_DATA)
        encoders = joblib.load(os.path.join("models", f'{MODEL}_{FOLD}_label_encoder.pkl'))
        cols = joblib.load(os.path.join("models", f'{MODEL}_{FOLD}_columns.pkl'))
        for c in cols:
            lbl = encoders[c]
            df.loc[:,c] = lbl.transform(df[c].values.tolist())

        clf = joblib.load(os.path.join("models", f'{MODEL}_{FOLD}.pkl'))
        
        
        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds 
        
    # predictions = predictions/5
    sub = pd.DataFrame(np.column_stack([test_idx,predictions]), columns = ['id','target'])
    return sub




if __name__ == "__main__":
    submission = predict()
    submission.to_csv (f'models/{MODEL}.csv',index = False)


