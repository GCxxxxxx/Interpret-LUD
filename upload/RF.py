# -*- coding: utf-8 -*-
"""
@author: Gcx
"""



import time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor as RF
from skopt import BayesSearchCV
from skopt.space import Integer


'''Enter the local path'''

#Input
phe_data_path ='' #full phenology data
training_data_path ='' #training data
test_data_path ='' #test data
tgdata_path = '' #mean temperature data
UM_para_path = ''#UM model
#Output
rf_para_path = '' #best_params
model_path = '' #model
training_pre = ''
test_pre = ''

#Training

randomseed = 11
specieslist = pd.Series(['Aesculus hippocastanum','Betula pendula','Fagus sylvatica','Larix decidua','Quercus robur','Sorbus aucuparia'])
feature_drop = ['spring_year','station','spring_doy','GDD','tc','te','pre']

rf_param=pd.DataFrame()
for i in range(len(specieslist)):
    name = specieslist[i]  
    for n in range(1,6):
        data = pd.read_csv(training_data_path+name+'_'+str(n)+'_train.csv')
        RF_reg = RF(random_state=randomseed,n_jobs=-1)
        bsearch = BayesSearchCV(
         RF_reg,
         {
          'max_depth':Integer(20,100),
          'min_samples_leaf':Integer(1,50),
          'min_samples_split':Integer(2,100),
         },
         n_iter=50,
         random_state=randomseed,
         cv=5,
         n_jobs=-1
         )

        GDD = data['GDD'].values
        climate = data.drop(feature_drop,axis=1,inplace = False)
        print(name,n,'GDD_go',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        bsearch.fit(climate,GDD)
        model_GDD = RF(random_state=randomseed,n_jobs=-1,**bsearch.best_params_)
        sp_param = pd.DataFrame([bsearch.best_params_])
        sp_param = sp_param.rename(index={0:name})
        rf_param = pd.concat([rf_param,sp_param], axis=0)
        model_GDD.fit(climate,GDD)
        joblib.dump(model_GDD, model_path+name+'_'+str(n)+'_rf.pkl')
        print(name,n,'GDD_done',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
    pass
pass
rf_param.to_csv(rf_para_path+'rf_hyperp_GDD.csv')


#Prediction

T_data = pd.read_csv(tgdata_path + 'tgdata.csv')
T_data['date'] = pd.to_datetime(T_data['day'])
T_data['year'] = T_data['date'].dt.year
T_data['month'] = T_data['date'].dt.month
T_data['day'] = T_data['date'].dt.day
T_data['doy'] = T_data['date'].dt.strftime('%j')
T_data['doy'] = T_data['doy'].astype(int)



for i in range(len(specieslist)):
    
    name = specieslist[i]
    
    para = pd.read_csv(UM_para_path + name + '_UN_Chuine2000_DA.csv')
    phedata = pd.read_csv(phe_data_path+name+'.csv')
    
    
    T = pd.DataFrame()
    t_dict = {str(phedata.loc[i, 'station']) + '_' + str(phedata.loc[i, 'spring_year']):
                  T_data.loc[T_data[(T_data['year'] == phedata.loc[i, 'spring_year'] - 1) & (T_data['month'] == 9) & (
                              T_data['day'] == 1)].index[0]:
                             T_data[(T_data['year'] == phedata.loc[i, 'spring_year'] - 1) & (T_data['month'] == 9) & (
                                         T_data['day'] == 1)].index[0] + 365,
                  [str(phedata.loc[i, 'station'])]].reset_index(drop=True).iloc[:, 0] for i in range(len(phedata))}
    T = pd.concat([T, pd.DataFrame(t_dict)], axis=1)
    T = T.fillna(method='pad', axis=0)
    
    
    
    for n in range(1,6):
        res_train = pd.DataFrame()
        res_test = pd.DataFrame()
        phe_train = pd.read_csv(training_data_path + name + '_'+str(n)+'_train.csv')
        phe_test = pd.read_csv(test_data_path + name + '_'+str(n)+'_test.csv')
        res_train['station'] = phe_train['station']
        res_train['year'] = phe_train['spring_year']
        res_train['spring'] = phe_train['spring_doy']
        res_train['GDD'] = phe_train['GDD']
        res_train['tc'] = phe_train['tc']+121
        res_test['station'] = phe_test['station']
        res_test['year'] = phe_test['spring_year']
        res_test['spring'] = phe_test['spring_doy']
        res_test['GDD'] = phe_test['GDD']
        res_test['tc'] = phe_test['tc']+121

        d = para.loc[n-1,'d']
        e = para.loc[n-1,'e']

    
        GDD_d = 1 / (1 + np.exp(d * (T - e)))
        
    
  
    
        train_x = phe_train.drop(feature_drop,axis=1,inplace = False)
        test_x = phe_test.drop(feature_drop,axis=1,inplace = False)
        



        model_GDD = joblib.load(model_path+name+'_'+str(n)+'_rf.pkl')
        pred_train_GDD = model_GDD.predict(train_x)
        pred_test_GDD = model_GDD.predict(test_x)
        res_train['pre_GDD'] = pred_train_GDD
        res_test['pre_GDD'] = pred_test_GDD
        
        
        for index in res_train.index:

            
            GDD = GDD_d[str(res_train.loc[index, 'station']) + '_' + str(res_train.loc[index, 'year'])].reset_index(
                drop=True)
                           
            
            R_a = 0
            for t in range(int(res_train.loc[index,'tc'] - 1), 365):
                R_a += GDD[t]
                if R_a >= res_train.loc[index,'pre_GDD']:
                    res_train.loc[index,'pre_spring'] = t - 121
                    break
                
                pass
            print(name,n,index)
        res_train.to_csv(training_pre+name+'_rf_train_'+str(n)+'.csv',index=False)
        
        
        for index in res_test.index:

            
            GDD = GDD_d[str(res_test.loc[index, 'station']) + '_' + str(res_test.loc[index, 'year'])].reset_index(
                drop=True)
                           
            
            R_a = 0
            for t in range(int(res_test.loc[index,'tc'] - 1), 365):
                R_a += GDD[t]
                if R_a >= res_test.loc[index,'pre_GDD']:
                    res_test.loc[index,'pre_spring'] = t - 121
                    break
                
                pass
            print(name,n,index)
        res_test.to_csv(test_pre+name+'_rf_test_'+str(n)+'.csv',index=False)
