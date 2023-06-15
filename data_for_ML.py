# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:42:07 2023

@author: Gcx
"""





import pandas as pd

#Input
phe_data_path = '' #full phenology data
UN_result_trainingset = ''
UN_result_testset = ''
#Output
training_data_path = ''
test_data_path = ''



specieslist = pd.Series(['Aesculus hippocastanum', 'Betula pendula', 'Fagus sylvatica', 'Larix decidua', 'Quercus robur','Sorbus aucuparia'])



for name in specieslist:    
    all_data = pd.read_csv(phe_data_path+name+'.csv')
    
    
    for n in range(1,6):
        
        phe_train = pd.read_csv(UN_result_trainingset+name+'_'+str(n)+'.csv')
        phe_test = pd.read_csv(UN_result_testset+name+'_'+str(n)+'.csv')
        
        phe_train_new = phe_train.merge(all_data,on=['station','spring_year','spring_doy'],how='left')
        phe_train_new.to_csv(training_data_path+name+'_'+str(n)+'_train.csv',index=False)
        
        phe_test_new = phe_test.merge(all_data,on=['station','spring_year','spring_doy'],how='left')
        phe_test_new.to_csv(test_data_path+name+'_'+str(n)+'_test.csv',index=False)
        
        print(name,n)
            
            

