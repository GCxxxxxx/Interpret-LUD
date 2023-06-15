# -*- coding: utf-8 -*-
"""
@author: Gcx
"""


import time
import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from sklearn.metrics import mean_squared_error as mse_score



'''Enter the local path'''

#Input
phe_data_path ='' #full phenology data
training_data_path ='' #training data
test_data_path ='' #test data
tgdata_path = '' #mean temperature data
lower_path = ''#the lower limit of the params
upper_path = ''#the upper limit of the params
#Output
UM_para_path = '' #model
training_pre = ''
test_pre = ''

#Training


T_data = pd.read_csv(tgdata_path+'tgdata.csv')
T_data['date'] = pd.to_datetime(T_data['day'])
T_data['year'] = T_data['date'].dt.year
T_data['month'] = T_data['date'].dt.month
T_data['day'] = T_data['date'].dt.day
T_data['doy'] = T_data['date'].dt.strftime('%j')
T_data['doy'] = T_data['doy'].astype(int)
lowerdata = pd.read_csv(lower_path+'pa_lower.csv')
upperdata = pd.read_csv(upper_path+'pa_upper.csv')

specieslist = pd.Series(['Aesculus hippocastanum', 'Betula pendula', 'Fagus sylvatica', 'Larix decidua', 'Quercus robur','Sorbus aucuparia'])


def UN(arg):
    a, b, c, d, e, w, z, t_c, C_crit = arg

    CA_d = 1 / (1 + np.exp(a * ((T - c) ** 2) + b * (T - c)))

    phe_C = pd.Series([182] * len(phe_use))  # from Sep. 1 of the preceding year

    phe_pre = pd.Series([366] * len(phe_use))

    GDD_d = 1 / (1 + np.exp(d * (T - e)))

    for index in phe_use.index:

        CA = CA_d[str(phe_use.loc[index, 'station']) + '_' + str(phe_use.loc[index, 'spring_year'])].reset_index(
            drop=True)
        C_a = 0
        for u in range(0, 365):
            C_a += CA[u]
            if C_a >= C_crit:
                phe_C[index] = u + 1
                break

        GDD = GDD_d[str(phe_use.loc[index, 'station']) + '_' + str(phe_use.loc[index, 'spring_year'])].reset_index(
            drop=True)
        t_c = int(t_c)
        t_all = int(t_c + phe_C[index])

        CA_all = CA[0:t_all].sum()
        F_crit = w * np.exp(z * CA_all)

        R_a = 0
        for t in range(phe_C[index] - 1, 365):
            R_a += GDD[t]
            if R_a >= F_crit:
                phe_pre[index] = t - 121
                break
            pass
    MSE = mse_score(phe_use['spring_doy'], phe_pre, squared=True)
    return MSE


result = pd.DataFrame(
    columns=['MSE', 'message', 'nfev', 'nit', 'success', 'a', 'b', 'c', 'd', 'e', 'w', 'z', 't_c', 'C_crit'])

print('ready go')

for i in range(0,6):

    name = specieslist[i]
    
    for n in range(1, 6):
        phedata = pd.read_csv(training_data_path + name + '_'+str(n)+'_train.csv')
        phe_use = phedata[['station', 'spring_year', 'spring_doy']].astype(int)
        T = pd.DataFrame()
        t_dict = {str(phedata.loc[i, 'station']) + '_' + str(phedata.loc[i, 'spring_year']):
                      T_data.loc[T_data[(T_data['year'] == phedata.loc[i, 'spring_year'] - 1) & (T_data['month'] == 9) & (
                                  T_data['day'] == 1)].index[0]:
                                 T_data[(T_data['year'] == phedata.loc[i, 'spring_year'] - 1) & (T_data['month'] == 9) & (
                                             T_data['day'] == 1)].index[0] + 365,
                      [str(phedata.loc[i, 'station'])]].reset_index(drop=True).iloc[:, 0] for i in range(len(phe_use))}
        T = pd.concat([T, pd.DataFrame(t_dict)], axis=1)
        T = T.fillna(method='pad', axis=0)
    
        lw = lowerdata[lowerdata['species'] == name].iloc[:, 1:].values
        lw = lw.squeeze()
        up = upperdata[upperdata['species'] == name].iloc[:, 1:].values
        up = up.squeeze()
    
        print(name, n, 'UN_Chuine2000,go', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        opt = dual_annealing(UN, bounds=list(zip(lw, up)), no_local_search=True)
        result.loc[n, 'MSE'] = opt.fun
        result.loc[n, 'message'] = opt.message
        result.loc[n, 'nfev'] = opt.nfev
        result.loc[n, 'nit'] = opt.nit
        result.loc[n, 'success'] = opt.success
        result.loc[n, 'a'] = opt.x[0]
        result.loc[n, 'b'] = opt.x[1]
        result.loc[n, 'c'] = opt.x[2]
        result.loc[n, 'd'] = opt.x[3]
        result.loc[n, 'e'] = opt.x[4]
        result.loc[n, 'w'] = opt.x[5]
        result.loc[n, 'z'] = opt.x[6]
        result.loc[n, 't_c'] = opt.x[7]
        result.loc[n, 'C_crit'] = opt.x[8]
        print(name, n, 'UN_Chuine2000, done', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    
    result.to_csv(UM_para_path + name + '_UN_Chuine2000_DA.csv',index=False)
    
    pass






T_data = pd.read_csv(tgdata_path+'tgdata.csv')
T_data['date'] = pd.to_datetime(T_data['day'])
T_data['year'] = T_data['date'].dt.year
T_data['month'] = T_data['date'].dt.month
T_data['day'] = T_data['date'].dt.day
T_data['doy'] = T_data['date'].dt.strftime('%j')
T_data['doy'] = T_data['doy'].astype(int)



for i in range(0,6):
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
    
    for n in range(1, 6):
        a = para.loc[n-1,'a']
        b = para.loc[n-1,'b']
        c = para.loc[n-1,'c']
        d = para.loc[n-1,'d']
        e = para.loc[n-1,'e']
        w = para.loc[n-1,'w']
        z = para.loc[n-1,'z']
        t_c = para.loc[n-1,'t_c']
        t_c = int(t_c)
        C_crit = para.loc[n-1,'C_crit']
        
       
        CA_d = 1 / (1 + np.exp(a * ((T - c) ** 2) + b * (T - c)))
        GDD_d = 1 / (1 + np.exp(d * (T - e)))
        
        #test data
        phedata1 = pd.read_csv(test_data_path+name+'_'+str(n)+'.csv')
        phe_use = phedata1[['station', 'spring_year', 'spring_doy']].astype(int)
        for index in phe_use.index:

            CA = CA_d[str(phe_use.loc[index, 'station']) + '_' + str(phe_use.loc[index, 'spring_year'])].reset_index(
                drop=True)
            C_a = 0
            for u in range(0, 365):
                C_a += CA[u]
                if C_a >= C_crit:
                    phe_use.loc[index,'tc'] = u + 1
                    break
                else:
                    phe_use.loc[index,'tc'] = 182
            GDD = GDD_d[str(phe_use.loc[index, 'station']) + '_' + str(phe_use.loc[index, 'spring_year'])].reset_index(
                drop=True)
            
            t_all = int(t_c + phe_use.loc[index,'tc'])
            phe_use.loc[index,'te'] = t_all
            CA_all = CA[0:t_all].sum()
            F_crit = w * np.exp(z * CA_all)
            phe_use.loc[index,'CA'] = CA_all
            GDD_a = 0
            for t0 in range(int(phe_use.loc[index,'tc'] - 1), int(phe_use.loc[index, 'spring_doy'] + 122)):
                GDD_a += GDD[t0]
                pass
            phe_use.loc[index,'GDD'] = GDD_a
                
            
            R_a = 0
            for t in range(int(phe_use.loc[index,'tc'] - 1), 365):
                R_a += GDD[t]
                if R_a >= F_crit:
                    phe_use.loc[index,'pre'] = t - 121
                    break
                
                pass
            print(name, n, index, 'test')
        phe_use['tc'] = phe_use['tc']-121
        phe_use['te'] = phe_use['te']-121
        phe_use['pre'] = phe_use['pre'].fillna(243)
        phe_use.to_csv(test_pre + name + '_'+str(n)+'.csv',index=False)


        #training data
        phedata2 = pd.read_csv(training_data_path+name+'_'+str(n)+'.csv')
        phe_use = phedata2[['station', 'spring_year', 'spring_doy']].astype(int)
        for index in phe_use.index:
        
            CA = CA_d[str(phe_use.loc[index, 'station']) + '_' + str(phe_use.loc[index, 'spring_year'])].reset_index(
                drop=True)
            C_a = 0
            for u in range(0, 365):
                C_a += CA[u]
                if C_a >= C_crit:
                    phe_use.loc[index,'tc'] = u + 1
                    break
                else:
                    phe_use.loc[index,'tc'] = 182
                    
            GDD = GDD_d[str(phe_use.loc[index, 'station']) + '_' + str(phe_use.loc[index, 'spring_year'])].reset_index(
                drop=True)
            
            t_all = int(t_c + phe_use.loc[index,'tc'])
            phe_use.loc[index,'te'] = t_all
            CA_all = CA[0:t_all].sum()
            F_crit = w * np.exp(z * CA_all)
            phe_use.loc[index,'CA'] = CA_all
            GDD_a = 0
            for t0 in range(int(phe_use.loc[index,'tc'] - 1), int(phe_use.loc[index, 'spring_doy'] + 122)):
                GDD_a += GDD[t0]
                pass
            phe_use.loc[index,'GDD'] = GDD_a
                
            
            R_a = 0
            for t in range(int(phe_use.loc[index,'tc'] - 1), 365):
                R_a += GDD[t]
                if R_a >= F_crit:
                    phe_use.loc[index,'pre'] = t - 121
                    break
                pass
            print(name, n, index,'train')
        phe_use['tc'] = phe_use['tc']-121
        phe_use['te'] = phe_use['te']-121
        phe_use['pre'] = phe_use['pre'].fillna(243)
        phe_use.to_csv(training_pre + name + '_'+str(n)+'.csv',index=False)


pass



