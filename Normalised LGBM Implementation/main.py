import DataLoading
import ModelTraining
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #df = DataLoading.DataSetLoadFolder("C:/Users/40237845/Documents/Ember_2017/ember_2017_2/train")
    test = DataLoading.DataSetLoadSingleFile("C:/Users/40237845/Documents/Ember_2017/ember_2017_2/test_features.jsonl")

    #print(df)
    #df["Histogram"] = pd.Categorical(df["Histogram"]).codes

    #X_train = df.iloc[:, 0:256]
    X_test = test.iloc[:, 0:256]
    #print(X)
    #y_train = df["Label"]
    y_test = test["Label"]
    #print(y)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #print(X_train)
    #model = ModelTraining.trainmodel(X_train, y_train)
    #model = ModelTraining.trainmodel(X, y)
    model = joblib.load("lgb2017.pkl")

    #y_pred = model.predict(X_test)
    #test_pred = model.predict_proba(X_test)
    #y_pred = (model.predict_proba(X_test)[:,1] >= 0.70).astype(bool)
    y_pred = model.predict(X_test)
    #print(test_pred)
    print(y_pred)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n\n', confusion)
    print('\nTrue Positives(TP) = ', confusion[0, 0])
    print('\nTrue Negatives(TN) = ', confusion[1, 1])
    print('\nFalse Positives(FP) = ', confusion[0, 1])
    print('\nFalse Negatives(FN) = ', confusion[1, 0])
    print(f"Overall Accuracy : {accuracy_score(y_test, y_pred)}")
    print(f"Precision Score : {precision_score(y_test, y_pred)}")
    print(f"Recall Score : {recall_score(y_test, y_pred)}")
    print(f"F1 Score : {f1_score(y_test, y_pred)}")
    ConfusionMatrixDisplay(confusion, display_labels=["Benign", "Malware"]).plot()
    plt.show()




