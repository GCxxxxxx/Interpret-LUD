# -*- coding: utf-8 -*-
"""
@author: Gcx
"""
   
    

import pandas as pd
from sklearn.model_selection import StratifiedKFold


'''Enter the local path'''

#Input
phe_data_path ='' #full phenology data
#Output
training_data_path = ''
test_data_path = ''






specieslist = pd.Series(['Aesculus hippocastanum','Betula pendula','Fagus sylvatica','Larix decidua','Quercus robur','Sorbus aucuparia'])
randomseed = 11

for name in specieslist:
    phedata = pd.read_csv(phe_data_path+name+'.csv')
    skf = StratifiedKFold(n_splits=5,random_state=randomseed,shuffle=True)
    X = phedata
    y = phedata['station']
    skf.split(X,y)
    n = 0
    trainlist = pd.DataFrame()
    testlist = pd.DataFrame()
    for train, test in skf.split(X, y):
        n += 1
        phe_train = phedata.iloc[train,:].reset_index(drop=True)
        phe_test = phedata.iloc[test,:].reset_index(drop=True)
        phe_train.to_csv(training_data_path+name+'_'+str(n)+'.csv',index=False)
        phe_test.to_csv(test_data_path+name+'_'+str(n)+'.csv',index=False)
        pass
    pass   
pass