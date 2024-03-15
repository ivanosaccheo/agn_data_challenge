import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import data_challenge_library as dcl
import matplotlib as mpl
from xgboost import XGBClassifier
import datetime



object_table = dcl.load_table(features = "important")
train, validation, test = dcl.prepare_sample(object_table, test_size = 0.001, validation_size = 0.2)

xgb = XGBClassifier()
xgb.fit(train.X, train.y)
importance_cuts = [0.1, 0.05, 0.01, 0.005, 0.003, 0.001,0.0005, 0.0003, 0.0002, 1e-4, 0.00005, 1e-5]
Niter = 30
accuracy_saver  = np.zeros((len(importance_cuts),Niter))
for i in range(Niter):
    print(f"Iteration: {i} over {Niter}")
    train, validation, test = dcl.prepare_sample(object_table, test_size = 0.001, validation_size = 0.2, shuffle_random_state=10*i,
                                                split_random_state_2=i*3)
    
    N_features = []
    accuracies = []
    for importance_cut in importance_cuts:
        logic = xgb.feature_importances_ >= importance_cut
        new_train_X = train.X[:,logic]
        new_validation_X = validation.X[:,logic]
        xgb_temp = XGBClassifier(max_depth=10)
        xgb_temp.fit(new_train_X, train.y)
        predictions = xgb_temp.predict(new_validation_X)
        accuracies.append(accuracy_score(validation.y, predictions))
        N_features.append(np.sum(logic))
    accuracy_saver[:,i] = accuracies

today = datetime.datetime.today()
save_directory = "plot_general"
save_name = f"features_importance_xgboost_{today}.png"

fig, ax = plt.subplots()
ax.errorbar(N_features, np.nanmean(accuracy_saver, axis =1), np.nanstd(accuracy_saver, axis =1), marker ='o')
ax.set_xlabel("N used features", fontsize = 15)
ax.set_ylabel("Accuracy", fontsize = 15)
plt.savefig(os.path.join(save_directory, save_name), bbox_inches = "tight")

for N, mean, std in zip(N_features, np.nanmean(accuracy_saver, axis =1), np.nanstd(accuracy_saver, axis =1)):
    print(N, mean, std)