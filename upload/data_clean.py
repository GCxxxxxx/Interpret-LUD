# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 08:34:31 2022

@author: Gcx
"""


import pandas as pd
import numpy as np
import time
import os

#Input
phe_data_path = ''
station_data_path = ''
#Output
coordinate_data_path = ''
cleandata_path = ''

def finddigit(str0):
    dig_sum = 0
    for strs in str0:
        if strs.isdigit():
            dig_sum+=1
            pass
        pass

    return dig_sum
        

def MAD_cal(phe_series):
    series_ab = abs(phe_series-np.median(phe_series))
    MAD = np.median(series_ab)
    return MAD



specieslist = pd.Series(['Aesculus hippocastanum','Betula pendula','Fagus sylvatica','Larix decidua','Quercus robur','Sorbus aucuparia'])

  

  
coor = pd.DataFrame()
coor_str = pd.DataFrame()
for j in range(len(specieslist)):
    name = specieslist[j]
    df = pd.read_csv(phe_data_path+name+'.csv')
    df = df.loc[df['spring_year']>1959].reset_index(drop=True)
    station_stat = df.groupby(['station']).count().reset_index(drop=False)
    stationlist = station_stat[station_stat['spring_year']>9]['station']
    stationlist = pd.DataFrame(stationlist)
    stationlist = stationlist.sort_values(by=['station'],ascending=True).reset_index(drop=True)
    
    phedata = stationlist.merge(df,on=['station'],how='left')
    
    
    coordinate = pd.read_table(station_data_path+name+'_station.dat',names=['station','longitude','latitude','elevation'],sep=',',encoding='utf-8')
    coor = pd.concat([coor,coordinate],axis=0,ignore_index=True)
    coordinate_str = pd.read_table(station_data_path+name+'_station.dat',names=['station','longitude','latitude','elevation'],sep=',',encoding='utf-8',dtype='str')
    coor_str = pd.concat([coor_str,coordinate_str],axis=0,ignore_index=True)
    pass
coor.drop_duplicates(inplace=True,ignore_index=True)
coor = coor.loc[:,['station','longitude','latitude']]

coor_str.drop_duplicates(inplace=True,ignore_index=True)
coor_str = coor_str.loc[:,['station','longitude','latitude']]

dup_coorlist = coor.loc[coor['station'].duplicated(),['station']].sort_values(by=['station'],ascending=True).reset_index(drop=True)
dup_coor = dup_coorlist.merge(coor,on = 'station',how = 'left')
coor_use = pd.DataFrame(columns=['station','longitude','latitude'])
stationdrop = pd.DataFrame(columns=['station'])

m=0
n=0
for i in range(len(dup_coorlist)):
    station = dup_coorlist.loc[i,'station']
    dup_temp = dup_coor.loc[dup_coor['station']==station].reset_index(drop=True).T
    dup_temp['diff'] = abs(dup_temp[0]-dup_temp[1])
    
    dup_temp_str = coor_str.loc[coor_str['station']==str(station)].reset_index(drop=True).T

    
    if max(dup_temp['diff'])>=0.1:
        stationdrop.loc[m,'station']=station
        m+=1
    else:
        long_len0 = finddigit(dup_temp_str.loc['longitude',0])
        long_len1 = finddigit(dup_temp_str.loc['longitude',1])
        lat_len0 = finddigit(dup_temp_str.loc['latitude',0])
        lat_len1 = finddigit(dup_temp_str.loc['latitude',1])
        coor_use.loc[n,'station']=station
        if long_len0>long_len1:
            coor_use.loc[n,'longitude'] = float(dup_temp_str.loc['longitude',0])
        elif long_len0<long_len1:
            coor_use.loc[n,'longitude'] = float(dup_temp_str.loc['longitude',1])
        else:
            coor_use.loc[n,'longitude'] = (dup_temp.loc['longitude',0]+dup_temp.loc['longitude',1])/2
            pass
        
        if long_len0>long_len1:
            coor_use.loc[n,'latitude'] = float(dup_temp_str.loc['latitude',0])
        elif long_len0<long_len1:
            coor_use.loc[n,'latitude'] = float(dup_temp_str.loc['latitude',1])
        else:
            coor_use.loc[n,'latitude'] = (dup_temp.loc['latitude',0]+dup_temp.loc['latitude',1])/2
            pass
        n+=1
        pass
    pass
coor = pd.concat([coor,dup_coorlist])
coor.drop_duplicates(subset='station',keep=False,inplace=True)
coor = pd.concat([coor,coor_use],ignore_index=True)
coor = coor.sort_values(by=['station'],ascending=True).reset_index(drop=True)
coor.to_csv(coordinate_data_path+'coordinate.csv',index=False)


'''
Extract the climate data based on the coordinate data, and remove the stations 
that were lacking the temperature data to get the new station list.
'''

stationlist0 = pd.read_csv('')#new station list


for name in specieslist:
    phe_all = pd.read_csv(phe_data_path+name+'.csv')
    stationlist = pd.DataFrame()
    stationlist['station'] = list(set(phe_all['station'])&set(stationlist0['station']))
    stationlist = stationlist.sort_values(by=['station'],ascending=True).reset_index(drop=True)
    phe_re = stationlist.merge(phe_all,how='left',on=['station'])
    phe_drop = pd.DataFrame(columns = ['station','spring_year'])
    for i in range(len(stationlist)):
        station = stationlist.iloc[i,0]
        phe_use = phe_all[phe_all.station == station].reset_index(drop=True)
        MAD_use = MAD_cal(phe_use.spring_doy)
        phe_drop0 = phe_use[(abs(phe_use.spring_doy-np.median(phe_use.spring_doy))/MAD_use)>2.5][['station','spring_year']]
        phe_drop = pd.concat([phe_drop,phe_drop0],axis=0,ignore_index=True)
        print(i,name)
        pass
    phe_final = pd.concat([phe_re,phe_drop],ignore_index=True)
    phe_final.drop_duplicates(subset=['station','spring_year'],keep=False,inplace=True)  
    phe_final.to_csv(cleandata_path+name+'.csv',index=False)
    pass
pass



