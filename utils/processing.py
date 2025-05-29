import numpy as np
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from datetime import timedelta
from .utils import time_lag

def process_data(tec, filter_name, wn, filter_functions):
    dict_tec = {}
    tec = pd.DataFrame(tec, columns = ['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
    # tec = tec[(tec['el'] > mask_cut_off) & (tec['time'] > 0)]
    
    for satellite, group in tec.groupby('satellite'):
        dict_tec[satellite] = group.values.tolist()
    
    for key in list(dict_tec.keys()):
        surrogate = []
        flag = 0
        data = dict_tec[key]
        
        for j in range(1, len(data)):
            if time_lag(data[j-1][0], data[j][0]) > wn * 1.7:
                surrogate.append(data[flag:j])
                flag = j
        
        if flag != 0:
            dict_tec[key] = surrogate + [data[flag:]]
        else:
            dict_tec[key] = [dict_tec[key]]
            
        filtered_sessions = []
        for session in dict_tec[key]: 
            if len(session) >= 100:
                if filter_name in filter_functions:
                    # session = butterfilt(session)
                    # session = cheb(session)
                    # session = remezfilt(session)
                    # session = firwinfilt(session)
                    # session = decimation(session)
                    session = filter_functions[filter_name](session, wn)
                filtered_sessions.append(session)
                
        dict_tec[key] = filtered_sessions
                
    return dict_tec
    
def process_data_path(dict_tec, filter_name, wn, filter_functions):
    filtered_dict_tec = {}
    
    for station, df in dict_tec.items():
        filtered_dict_tec[station] = {}
    
        for satellite_id, group in df.groupby('satellite'):
            data_list = group.values.tolist()
            
            if satellite_id not in filtered_dict_tec[station]:
                filtered_dict_tec[station][satellite_id] = []
                # filtered_dict_tec[station][satellite_id].append(['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
            filtered_dict_tec[station][satellite_id].extend(data_list)

    for key1 in filtered_dict_tec.keys():
        for key2 in filtered_dict_tec[key1].keys():
            surrogate = []
            flag = 0
            data = filtered_dict_tec[key1][key2][:-10]
            
            for j in range(1, len(data)):
                if time_lag(data[j-1][0], data[j][0]) > wn * 1.7:
                    surrogate.append(data[flag:j])
                    flag = j
            
            if flag != 0:
                filtered_dict_tec[key1][key2] = surrogate + [data[flag:]]
            else:
                filtered_dict_tec[key1][key2] = [filtered_dict_tec[key1][key2]]
            
    for key1 in filtered_dict_tec.keys():
        for key2 in filtered_dict_tec[key1].keys():
            
            filtered_sessions = []
            for session in filtered_dict_tec[key1][key2]: 
                if len(session) >= 100:
                    # print(session)
                    if filter_name in filter_functions:
                        try:
                            # session = butterfilt(session)
                            # session = cheb(session)
                            # session = remezfilt(session)
                            # session = firwinfilt(session)
                            # session = decimation(session)
                            session = filter_functions[filter_name](session, wn)
                        except: ZeroDivisionError
                    filtered_sessions.append(session)
                    
            filtered_dict_tec[key1][key2] = filtered_sessions
                
    return filtered_dict_tec

# Сохранение данных (.csv файл для отдельного прохода где только время и ТЕС, тут сломано, надо проверять и переделывать)

def save_data(dict_tec, output_directory, file_path,ds, satellite_num, graph_tec, filter_name):
    time_surr, tec_surr = [],[]
    for t in list(dict_tec.keys()):
        for count in range(len(dict_tec[t])):
            # if filter_name in filter_functions:
            #     dict_tec[t][count] = filter_functions[filter_name](dict_tec[t][count])    
            time, tec = graph_tec(dict_tec, t, count)
            time_surr.extend(time)
            tec_surr.extend(tec)
    day_download = ds.strftime('%d-%B-%Y')              
    data = pd.DataFrame({'time': time_surr, 'tec': tec_surr})
    mean_value = data.groupby('time')['tec'].mean().reset_index()
    data = data.drop(columns = ['tec']).drop_duplicates(subset=['time']).merge(mean_value, on ='time')  
    data = data.sort_values(by='time').reset_index(drop=True)
    file_name = os.path.basename(file_path)
    filepath = os.path.dirname(output_directory + '/')
    csv_path = os.path.join(filepath, f'processed_tec{file_name[:4]}{satellite_num}-{day_download}-{filter_name}.csv')
    return data.to_csv(csv_path, index = False, sep =' ', na_rep = '')