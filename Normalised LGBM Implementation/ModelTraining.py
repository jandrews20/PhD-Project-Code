import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def trainmodel(X_train, y_train):

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    model.booster_.save_model("model.txt")
    joblib.dump(model, 'lgb.pkl')

    return model