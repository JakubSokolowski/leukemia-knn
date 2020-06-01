import sys
import pandas as pd
from scipy import stats
import numpy as np


sys.path.append('C:/Users/writh/Desktop/leukemia-knn')
from src.reader.columns import symptoms_name_dict
from src.reader.dir import LEUKEMIA_MASTER_PATH

leukimia_full = pd.read_csv(LEUKEMIA_MASTER_PATH)
leukimia_full.rename(columns=symptoms_name_dict, inplace=True)


feature_mean_arr = [0] * 20
all_classes = []

for i in range(20):
    data_of_type = leukimia_full.loc[leukimia_full['class']==i+1]
    all_classes.append(data_of_type.iloc[:, 0:20])



for i in range(0,20): # class[i]
    class1 = all_classes[i]
    #print('klasa i ', i)

    other_classes = list(range(0, 20))
    other_classes.remove(i)
    for j in other_classes: # classes that are not i
        class2 = all_classes[j]
        #print('klasa j ', j)

        for feature in range(0,20): # all features
            y = stats.ks_2samp( class1.iloc[:, feature] , class2.iloc[:, feature] )
            feature_mean_arr[feature] += y.statistic

            #print('cecha ', feature)
        

for i in range(0, 20):
    feature_mean_arr[i] = feature_mean_arr[i] / 20

print( pd.DataFrame(feature_mean_arr).sort_values(by=[0], ascending=False) )
