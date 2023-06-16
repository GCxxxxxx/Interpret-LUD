# -*- coding: utf-8 -*-
"""
@author: Gcx
"""




import numpy as np
import time
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
import pandas as pd
import joblib
from interpret.glassbox import ExplainableBoostingRegressor





'''Enter the local path'''

#Input
phe_data_path ='' #full phenology data
training_data_path ='' #training data
test_data_path ='' #test data
tgdata_path = '' #mean temperature data
UM_para_path = ''#UM model
#Output
model_path = '' #model
training_pre = ''
test_pre = ''
imp_path = ''
interpret_path = ''


#Training
randomseed = 11
specieslist = pd.Series(['Aesculus hippocastanum','Betula pendula','Fagus sylvatica','Larix decidua','Quercus robur','Sorbus aucuparia'])
feature_drop = ['spring_year','station','spring_doy','GDD','tc','te','pre']

for i in range(len(specieslist)):
    name = specieslist[i]
    for n in range(1,6):
        print(name,n,'go',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        data = pd.read_csv(training_data_path+name+'_'+str(n)+'_train.csv')
        X = data.drop(feature_drop,axis=1,inplace = False)
        y = data['GDD'].values
        ebm = ExplainableBoostingRegressor(random_state=randomseed,n_jobs=-1,validation_size=0.2)
        ebm.fit(X, y)
        joblib.dump(ebm, model_path+name+'_'+str(n)+'_ebm.pkl') #save the model
        print(name,n,'done',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        pass
    pass
pass


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
        


        model_GDD = joblib.load(model_path+name+'_'+str(n)+'_ebm.pkl')
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
        res_train.to_csv(training_pre+name+'_ebm_train_'+str(n)+'.csv',index=False)
        
        
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
        res_test.to_csv(test_pre+name+'_ebm_test_'+str(n)+'.csv',index=False)
              
#EBM importance
for i in range(len(specieslist)):
    name = specieslist[i]
    importance = pd.DataFrame()
    importance_trans = pd.DataFrame()
    for n in range(1,6):
        model_importance = pd.DataFrame()
        importance_ad = pd.DataFrame()
        ebm_model = joblib.load(model_path+name+'_'+str(n)+'_ebm.pkl')
        ebm_global = ebm_model.explain_global()
        model_importance['feature'] = ebm_global.data()['names']
        model_importance['imp'+str(n)] = ebm_global.data()['scores']
        ad = model_importance['imp'+str(n)].sum()
        importance_ad['feature'] = model_importance['feature']
        importance_ad['imp'+str(n)] = model_importance['imp'+str(n)]/ad
        if n == 1:
            importance = pd.concat([importance, model_importance], axis=1)
            importance_trans = pd.concat([importance_trans, importance_ad], axis=1)
        else:
            importance = pd.merge(importance, model_importance, on='feature',how='outer')
            importance_trans = pd.merge(importance_trans, importance_ad, on='feature',how='outer')
        pass
    importance['mean_imp'] = importance.mean(axis=1,numeric_only=True)
    importance['min_imp'] = importance.min(axis=1, numeric_only=True)
    importance['max_imp'] = importance.max(axis=1, numeric_only=True)
    importance_trans['mean_imp'] = importance_trans.mean(axis=1, numeric_only=True)
    importance_trans['min_imp'] = importance_trans.min(axis=1, numeric_only=True)
    importance_trans['max_imp'] = importance_trans.max(axis=1, numeric_only=True)
    importance.to_csv(imp_path+name+'_ebm_or.csv',index=False)
    importance_trans.to_csv(imp_path + name + '_ebm.csv', index=False)
   

#Interpreting
data = pd.read_csv(training_data_path+'Aesculus hippocastanum_1_train.csv')
climate = data.drop(feature_drop,axis=1,inplace = False)
feature_list = pd.DataFrame(columns=['feature'],data=climate.columns)
for i in range(len(specieslist)):    
    name = specieslist[i]
    for k in range(len(feature_list)):
        eff = pd.DataFrame()
        feature = feature_list.loc[k,'feature']     
        for n in range(1,6):
            eff_0 = pd.DataFrame()
            ebm_model = joblib.load(model_path+name+'_'+str(n)+'_ebm.pkl')
            ebm_global = ebm_model.explain_global()
            f_list = ebm_global.data()['names']
            eff_0[feature+'_x'+str(n)] = ebm_global.data(f_list.index(feature))['names'][:-1]
            eff_0[feature+'_y'+str(n)] = ebm_global.data(f_list.index(feature))['scores']
            eff = pd.concat([eff,eff_0],axis=1,join='outer')    
            pass
        eff.to_csv(interpret_path+name+'_'+feature+'_ebm.csv',index=False)
        print(i,k, eff.isnull().sum(axis=1).sum())
