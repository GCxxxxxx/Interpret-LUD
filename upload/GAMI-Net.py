# -*- coding: utf-8 -*-
"""
@author: Gcx
"""




import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import tensorflow as tf
from gaminet import GAMINet
from gaminet.utils import local_visualize
from gaminet.utils import global_visualize_density
from gaminet.utils import feature_importance_visualize
from gaminet.utils import plot_trajectory
from gaminet.utils import plot_regularization


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
random_state = 11
specieslist = pd.Series(['Aesculus hippocastanum','Betula pendula','Fagus sylvatica','Larix decidua','Quercus robur','Sorbus aucuparia'])
feature_drop = ['spring_year','station','spring_doy','GDD','tc','te','pre']





def metric_wrapper(metric, scaler):
    def wrapper(label, pred):
        return metric(label, pred, scaler=scaler)
    return wrapper

def rmse(label, pred, scaler):
    pred = scaler.inverse_transform(pred.reshape([-1, 1]))
    label = scaler.inverse_transform(label.reshape([-1, 1]))
    return np.sqrt(np.mean((pred - label)**2))
def get_phe(name,n):
   task_type = 'Regression'
   
   data_train = pd.read_csv(training_data_path + name + '_' + str(n) + '_train.csv')
   data_test = pd.read_csv(test_data_path + name + '_' + str(n) + '_test.csv')
   
   data_all0 = pd.concat([data_train,data_test],axis=0,ignore_index=True)
   data_all = data_all0.drop(feature_drop,axis=1,inplace = False)
   meta_info = {data_all.columns[i]:{'type':'continuous'} for i in range(len(data_all.columns))}
   meta_info.update({'GDD':{'type':'target'}})
   x0 = data_all.iloc[:,:].values
   y0 = data_all0.loc[:,['GDD']].values
   x_train_data, y_train_data = data_train.drop(feature_drop,axis=1,inplace = False).iloc[:,:].values, data_train.loc[:,['GDD']].values
   x_test_data, y_test_data = data_test.drop(feature_drop,axis=1,inplace = False).iloc[:,:].values, data_test.loc[:,['GDD']].values
   
   
   
   x_train = np.zeros((x_train_data.shape[0], x_train_data.shape[1]), dtype=np.float32)
   x_test = np.zeros((x_test_data.shape[0], x_test_data.shape[1]), dtype=np.float32)
   for i, (key, item) in enumerate(meta_info.items()):
       if item['type'] == 'target':
           sy = MinMaxScaler((0, 1))
           y_scaler = sy.fit(y0)
           y_train = y_scaler.transform(y_train_data)
           y_test = y_scaler.transform(y_test_data)
           meta_info[key]['scaler'] = sy
       elif item['type'] == 'categorical':
           enc = OrdinalEncoder()
           x_fit = enc.fit(x0[:,[i]])
           x_train[:,[i]] = x_fit.transform(x_train_data[:,[i]])
           x_test[:,[i]] = x_fit.transform(x_test_data[:,[i]])
           meta_info[key]['values'] = []
           for item in enc.categories_[0].tolist():
               try:
                   if item == int(item):
                       meta_info[key]['values'].append(str(int(item)))
                   else:
                       meta_info[key]['values'].append(str(item))
               except ValueError:
                   meta_info[key]['values'].append(str(item))
       else:
           sx = MinMaxScaler((0, 1))
           x_fit = sx.fit(x0[:,[i]])
           x_train[:,[i]] = x_fit.transform(x_train_data[:,[i]])
           x_test[:,[i]] = x_fit.transform(x_test_data[:,[i]])
           meta_info[key]['scaler'] = sx 
  
   
   return x_train, x_test, y_train,y_test, task_type, meta_info, metric_wrapper(rmse,sy) 



for i in range(len(specieslist)):
    name = specieslist[i]

        
    for n in range(1,6):
        x_train, x_test, y_train,y_test, task_type, meta_info, get_metric = get_phe(name,n)
        ## Note the current GAMINet API requires input features being normalized within 0 to 1.
        model = GAMINet(interact_num=20, meta_info=meta_info,
                        interact_arch=[40] * 5, subnet_arch=[40] * 5, 
                        batch_size=3456, task_type=task_type, activation_func=tf.nn.relu, 
                        main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500, 
                        lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                        heredity=True, loss_threshold=0.01, reg_clarity=1,
                        mono_increasing_list=[], mono_decreasing_list=[], ## the indices list of features
                        verbose=False, val_ratio=0.2, random_state=random_state)
        print(name,n,'go',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        model.fit(x_train, y_train)
    
        val_x = x_train[model.val_idx, :]
        val_y = y_train[model.val_idx, :]
        tr_x = x_train[model.tr_idx, :]
        tr_y = y_train[model.tr_idx, :]
        pred_train = model(tr_x)
        pred_val = model(val_x)
        pred_test = model(x_test)
        pred_train = np.array(pred_train)
        pred_val = np.array(pred_val)
        pred_test = np.array(pred_test)
        
        
        
        
        
        
        simu_dir = model_path
        model.save(folder=model_path, name=name+'_'+str(n)+'_gami')
        gaminet_stat = np.hstack([np.round(get_metric(tr_y, pred_train),5), 
                              np.round(get_metric(val_y, pred_val),5),
                              np.round(get_metric(y_test, pred_test),5)])
        del model
        print(gaminet_stat)
        print(name,n,'done',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))



T_data = pd.read_csv(tgdata_path + 'tgdata.csv')
T_data['date'] = pd.to_datetime(T_data['day'])
T_data['year'] = T_data['date'].dt.year
T_data['month'] = T_data['date'].dt.month
T_data['day'] = T_data['date'].dt.day
T_data['doy'] = T_data['date'].dt.strftime('%j')
T_data['doy'] = T_data['doy'].astype(int)


specieslist = pd.Series(['Aesculus hippocastanum','Betula pendula','Fagus sylvatica','Larix decidua','Quercus robur','Sorbus aucuparia'])
feature_drop = ['spring_year','station','spring_doy','GDD','tc','te','pre']



def get_phe(name,n):
   task_type = 'Regression'

   data_train = pd.read_csv(training_data_path + name + '_' + str(n) + '_train.csv')
   data_test = pd.read_csv(test_data_path + name + '_' + str(n) + '_test.csv')
   
   data_all0 = pd.concat([data_train,data_test],axis=0,ignore_index=True)
   data_all = data_all0.drop(feature_drop,axis=1,inplace = False)
   meta_info = {data_all.columns[i]:{'type':'continuous'} for i in range(len(data_all.columns))}
   meta_info.update({'GDD':{'type':'target'}})
   x0 = data_all.iloc[:,:].values
   y0 = data_all0.loc[:,['GDD']].values
   x_train_data, y_train_data = data_train.drop(feature_drop,axis=1,inplace = False).iloc[:,:].values, data_train.loc[:,['GDD']].values
   x_test_data, y_test_data = data_test.drop(feature_drop,axis=1,inplace = False).iloc[:,:].values, data_test.loc[:,['GDD']].values
   
   
   
   x_train = np.zeros((x_train_data.shape[0], x_train_data.shape[1]), dtype=np.float32)
   x_test = np.zeros((x_test_data.shape[0], x_test_data.shape[1]), dtype=np.float32)
   for i, (key, item) in enumerate(meta_info.items()):
       if item['type'] == 'target':
           sy = MinMaxScaler((0, 1))
           y_scaler = sy.fit(y0)
           y_train = y_scaler.transform(y_train_data)
           y_test = y_scaler.transform(y_test_data)
           meta_info[key]['scaler'] = sy
       elif item['type'] == 'categorical':
           enc = OrdinalEncoder()
           x_fit = enc.fit(x0[:,[i]])
           x_train[:,[i]] = x_fit.transform(x_train_data[:,[i]])
           x_test[:,[i]] = x_fit.transform(x_test_data[:,[i]])
           meta_info[key]['values'] = []
           for item in enc.categories_[0].tolist():
               try:
                   if item == int(item):
                       meta_info[key]['values'].append(str(int(item)))
                   else:
                       meta_info[key]['values'].append(str(item))
               except ValueError:
                   meta_info[key]['values'].append(str(item))
       else:
           sx = MinMaxScaler((0, 1))
           x_fit = sx.fit(x0[:,[i]])
           x_train[:,[i]] = x_fit.transform(x_train_data[:,[i]])
           x_test[:,[i]] = x_fit.transform(x_test_data[:,[i]])
           meta_info[key]['scaler'] = sx 
  
   
   return x_train, x_test, y_train,y_test,data_train,data_test,y_scaler





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
        x_train, x_test, y_train,y_test,data_train,data_test,y_scaler = get_phe(name,n)
        model = GAMINet(meta_info={})
        model.load(folder= model_path, name=name+'_'+str(n)+'_gami')
        
        
        
        d = para.loc[n-1,'d']
        e = para.loc[n-1,'e']

    
        GDD_d = 1 / (1 + np.exp(d * (T - e)))
        
        
        res_train = pd.DataFrame()
        res_test = pd.DataFrame()
        
        res_train['station'] = data_train['station']
        res_train['year'] = data_train['spring_year']
        res_train['spring'] = data_train['spring_doy']
        res_train['tc'] = data_train['tc']+121
        res_test['station'] = data_test['station']
        res_test['year'] = data_test['spring_year']
        res_test['spring'] = data_test['spring_doy']
        res_test['tc'] = data_test['tc']+121
        
        pred_train = model(x_train)
        pred_test = model(x_test)
        pred_train = np.array(pred_train)
        pred_test = np.array(pred_test)
        
        train_data = y_scaler.inverse_transform(pred_train.reshape([-1, 1]))
        test_data = y_scaler.inverse_transform(pred_test.reshape([-1, 1]))
        train_or = y_scaler.inverse_transform(y_train.reshape([-1, 1]))
        test_or = y_scaler.inverse_transform(y_test.reshape([-1, 1]))
        
        res_train['GDD'] = train_or
        res_train['pre_GDD'] = train_data
        res_test['GDD'] = test_or
        res_test['pre_GDD'] = test_data
        
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
        res_train.to_csv(training_pre+name+'_gami_train_'+str(n)+'.csv',index=False)
        
        
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
        res_test.to_csv(test_pre+name+'_gami_test_'+str(n)+'.csv',index=False)


#GAMI-Net importance
for i in range(len(specieslist)):    
    name = specieslist[i]
    importance = pd.DataFrame()
    for n in range(1,6):
        model_importance = pd.DataFrame()
        importance_ad = pd.DataFrame()
        model = GAMINet(meta_info={})
        model.load(folder=model_path, name=name+'_'+str(n)+'_gami')
        data_dict_global = model.global_explain(save_dict=False) 
        model_importance['feature'] = data_dict_global.keys()
        for k in range(len(model_importance['feature'])):
            model_importance.loc[k,'imp'+str(n)] = data_dict_global[model_importance.loc[k,'feature']]['importance']
            pass
        if n == 1:
            importance = pd.concat([importance, model_importance], axis=1)
        else:
            importance = pd.merge(importance, model_importance, on='feature',how='outer')
        pass
    importance['mean_imp'] = importance.mean(axis=1,numeric_only=True)
    importance['min_imp'] = importance.min(axis=1, numeric_only=True)
    importance['max_imp'] = importance.max(axis=1, numeric_only=True)
    importance.to_csv(imp_path+name+'_gami.csv',index=False)



#Interpreting
data = pd.read_csv(training_data_path+'Aesculus hippocastanum_1_train.csv')
climate = data.drop(feature_drop,axis=1,inplace = False)
feature_list = pd.DataFrame(columns=['feature'],data=climate.columns)

def get_diff(name,n):
   
   data_train = pd.read_csv(training_data_path + name + '_'+str(n)+'_train.csv')
   data_test = pd.read_csv(test_data_path + name + '_'+str(n)+'_test.csv')
   data_all0 = pd.concat([data_train,data_test],axis=0,ignore_index=True)  
   y0 = data_all0.loc[:,['GDD']].values
   diff = y0.max()-y0.min() 
   return diff

for i in range(len(specieslist)):    
    name = specieslist[i]
    
    for k in range(len(feature_list)):
        eff = pd.DataFrame()
        feature = feature_list.loc[k,'feature'] 
        for n in range(1,6):
            eff_0 = pd.DataFrame()
            model = GAMINet(meta_info={})
            model.load(folder=model_path, name=name+'_'+str(n)+'_gami')
            diff = get_diff(name,n)
            data_dict_global = model.global_explain(save_dict=False)
            eff_0[feature+'_x'+str(n)] = data_dict_global[feature]['inputs']
            eff_0[feature+'_y'+str(n)] = data_dict_global[feature]['outputs']*diff
            eff = pd.concat([eff,eff_0],axis=1,join='outer')
            pass
        eff.to_csv(interpret_path+name+'_'+feature+'_gami.csv',index=False)
        print(i,k, eff.isnull().sum(axis=1).sum())
