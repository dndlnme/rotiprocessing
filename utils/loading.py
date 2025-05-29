
import os
from pathlib import Path
import pandas as pd
from datetime import timedelta
from .utils import addtime 

def load_data(file_path, mask_cut_off, time_ceil, time_floor, satellite_num, stations):
    with open(file_path, 'r') as tec:
        tec = tec.readlines()
    tec = tec[5:]
    for i in range(len(tec)):
        tec[i] = list(map(float, tec[i].split()))[1:-1]    
    tec = pd.DataFrame(tec, columns=['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
    tec = tec[(tec['el'] > mask_cut_off) & (tec['time'] > time_floor) & (tec['time'] < time_ceil) & (tec['vtec'] > 0)]
    if satellite_num in tec['satellite'].tolist():
        tec = tec[tec['satellite'] == satellite_num]
    tec = tec.values.tolist()
    return tec      

def load_data_path(data_path, mask_cut_off, time_ceil, time_floor, satellite_num, stations):
    dict_tec = {}
    p = Path(data_path)

    if not p.is_dir():
        raise ValueError(f"The path {data_path} is not a valid directory.")

    for station in stations:
        period_tec = []
        time_bias = timedelta(hours=0)
        
        found_data = False
        for CMN in sorted(os.listdir(data_path)):
            if (CMN.startswith(station) or CMN.startswith(station.lower())):
                found_data = True
                CMN_path = p / CMN
                
                try:
                    daily_data = load_data(CMN_path, mask_cut_off, time_ceil, time_floor, satellite_num, stations)
                except Exception as e:
                    print(f"Error loading data from {CMN_path}: {e}")
                    continue
                
                for entry in daily_data:
                    time_entry = entry[0] + time_bias.total_seconds() / 3600
                    new_entry = [time_entry] + entry[1:]
                    period_tec.append(new_entry)
                
                time_bias += timedelta(hours=24)

        if found_data:
            dict_tec[station.upper()] = pd.DataFrame(period_tec, columns=['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])

    return dict_tec

def load_bz_data(bz_path, DoYs, DoYe):
    with open(bz_path, 'r') as bz:
        bz = bz.readlines()
    for i in range(len(bz)):
        bz[i] = list(map(float, bz[i].split()))  
    bz = pd.DataFrame(bz)
    bz = bz.filter(bz.columns[[0,1,2,3,13,16,18]])
    bz.columns = ['year', 'DOY', 'hour', 'minute', 'INF', 'bz(GSE)', 'bz(GSM)']
    bz = bz[(bz['bz(GSE)'] < 9999) & (bz['bz(GSM)'] < 9999)]
    bz['time'] = pd.to_timedelta(bz['hour'], unit='h') + pd.to_timedelta(bz['minute'], unit='m')
    bz['TIME'] = bz['time'].dt.components.apply(lambda x: f"{int(x.hours):02}:{int(x.minutes):02}", axis=1)
    bz['TIME_DECIMAL'] = bz['hour'] + bz['minute']/60
    bz = bz[(bz['DOY'] <= DoYe) & (bz['DOY'] >= DoYs)]
    bz = addtime(bz)
    
    return bz

def load_iaga_data(ae_path, DoYs, DoYe):
    ae = pd.read_csv(ae_path,  sep='\s+', skiprows = 14)
    ae.drop('|', axis = 1, inplace = True)
    ae['TIME'] = pd.to_datetime(ae['TIME'], format='%H:%M:%S.%f')
    ae['TIME_UTC'] = pd.to_datetime(ae['TIME'], format='%H:%M:%S.%f').dt.strftime('%H:%M')
    ae['TIME_DECIMAL'] = ae['TIME'].dt.hour + ae['TIME'].dt.minute / 60
    ae.drop('TIME', axis = 1, inplace = True)
    ae = ae[(ae['DOY'] <= DoYe) & (ae['DOY'] >= DoYs)]
    ae = addtime(ae)
    return ae 
