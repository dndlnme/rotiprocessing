import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import pandas as pd
from scipy.signal import  welch
import piecewise_regression
import scaleogram as scg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
from .utils import datetime2decimals, discret_detect, geographic_to_magnetic_latitude, decimation_2, butterfilt, cheb, remezfilt, firwinfilt, decimation, convert_time_to_mlt, ZP_oval, time_lag, norm_pmi, joint_prob, extend_edges, probability_from_histogram

filter_functions = {
    'butter': butterfilt,
    'cheby': cheb,
    'remez': remezfilt,
    'firwin': firwinfilt,
    'decimation': decimation
}
coikw = {'alpha': 0.5, 'hatch': '/'}  

def graph_roti(dict_tec, t, count):
    rot, time, latitude = [], [], []
    dict_tec1 = pd.DataFrame(copy.copy(dict_tec[t][count]), columns=['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
    dict_tec1 = dict_tec1.drop_duplicates(subset=['time']).sort_values(by='time').reset_index(drop=True)
    # fs = 1 / (np.mean(np.diff(dict_tec1['time'])) * 3600)
    try:
        fs = 1 / (60 / discret_detect(dict_tec1['time'].tolist()[0], dict_tec1['time'].tolist()[1]))
        # wn_roti = 1 / 61
        # dict_tec1['vtec'] = lowpass_filter(dict_tec1['vtec'], fs, wn_roti, order = 5)
        # dict_tec1[t][count] = dict_tec1[t][count][::discret_detect(dict_tec1[t][count][0][0], dict_tec1[t][count][1][0])]
        dict_tec1 = dict_tec1.iloc[::int(round(60 * fs))].reset_index(drop=True) 
        rot = np.abs(dict_tec1['vtec'].diff().iloc[1:])  
        time = dict_tec1['time'].iloc[1:] 
        latitude = dict_tec1['lat'].iloc[1:]
        longitude = dict_tec1['lon'].iloc[1:] 
        
        roti, time_mean, latitude_mean, longitude_mean = [], [], [], []

        for i in range(0, len(rot) - 4):
            roti.append(np.std(rot[i:i+5]))
            time_mean.append(np.median(time.iloc[i: i+5])) 
            latitude_mean.append(np.median(latitude.iloc[i:i+5]))
            longitude_mean.append(np.median(longitude.iloc[i:i+5]))
        if len(time_mean) != len(roti):
            raise ValueError("Length of time_mean and roti must be the same")
    except ZeroDivisionError:
        print(f'{t} broken satellite')
        time_mean, roti, latitude_mean, longitude_mean = [], [], [], []
    return [time_mean, roti, latitude_mean, longitude_mean]

# Выделение ТЕС и времени и построение графиков ТЕС, ROTI

def graph_tec(dict_tec, t, count):
    df = pd.DataFrame(dict_tec[t][count], columns = ['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
    time = df['time'].tolist()
    vstec = df['vtec'].tolist()
           
    return [time, vstec]

def plot_graph_tec_path(dict_tec, ds):
        for key in dict_tec.keys():
            tec = dict_tec[key]
            plot_graph_tec(tec, ds)

def plot_graph_roti(dict_tec, ds):
    graph_num = sum(len(value) for value in dict_tec.values())
    colors = [plt.get_cmap('gist_ncar')(float(i / graph_num)) for i in range(graph_num)]

    fig, ax = plt.subplots(figsize=(16, 9))
    count_index = 0
    for t in list(dict_tec.keys()):
        for count in range(len(dict_tec[t])):
            time, roti, latitude, _ = graph_roti(dict_tec, t, count)
            ax.plot(time, roti, c = colors[count_index], linewidth = 0.63)
            count_index += 1
            
    ax.set_xlabel('UTC, часы')
    ax.set_ylabel('ROTI, TECU/мин')
    # plt.title(f'ROTI {ds}')
    plt.tight_layout()
    plt.show()

def plot_graph_roti_path(dict_tec, ds):
    for key in dict_tec.keys():
        plot_graph_roti(dict_tec[key], ds)
    
def plot_graph_tec(dict_tec, ds):
    graph_num = sum(len(value) for value in dict_tec.values())
    colors = [plt.get_cmap('gist_ncar')(float(i / graph_num)) for i in range(graph_num)]

    fig, ax = plt.subplots(figsize=(16, 9))
    count_index = 0
    for t in list(dict_tec.keys()):
        for count in range(len(dict_tec[t])):
            time, tec = graph_tec(dict_tec, t, count)
            ax.plot(time, tec, c = colors[count_index], linewidth = 0.63, label = f'{t} satellite {count+1} session')
            count_index += 1
            # plt.text(time[0], tec[0], f'{t} satellite, {count+1} session', 
            # fontsize = 6)
            
    ax.set_xlabel('UTC, часы')
    ax.set_ylabel('TEC, TECu')
    # plt.title(f'TEC {ds}')
    # plt.legend()
    if len(dict_tec) < 25:
        plt.legend(loc = 'upper center', fontsize = 'xx-small', bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    plt.show() 
    
# Спектры ТЕС через Уэлча (для каждого прохода отдельно)
    
def plot_tec_spectrogram(dict_tec, key):
    graph_num = sum(len(value) for value in dict_tec.values())
    colors = [plt.get_cmap('gist_ncar')(float(i / graph_num)) for i in range(graph_num)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    count_index = 0
    Nsub = 10
    for t in list(dict_tec.keys()):
        for count in range(len(dict_tec[t])):
            time, vstec = graph_tec(dict_tec, t, count)
            # Nfft = len(vstec) - 1
            nperseg = len(vstec) / Nsub
            if len(time) > 100:
                sampling_freq = 1 / (60 / discret_detect(time[1], time[2]))
                f, Pxx_den = welch(vstec, fs = sampling_freq, nperseg = len(vstec)/Nsub, noverlap=nperseg/2, scaling='spectrum', nfft=nperseg, window= ('kaiser', 14), detrend = 'linear')
                if len(dict_tec) <= 15:
                    try:
                        pw_fit = piecewise_regression.Fit(np.log10(f[1:]), np.log10(Pxx_den[1:]), n_breakpoints=1)
                        # pw_fit.plot()
                        print(f'{t} satellite {count+1} session')
                        pw_results = pw_fit.get_results()
                        print('alpha1' ,pw_results['estimates']['alpha1'])
                        print('breakpoint' ,pw_results['estimates']['breakpoint1'])
                    except TypeError:
                        print('Linear regression failed')
                # f, Pxx_den = welch(vstec, fs = sampling_freq, noverlap=0, nperseg = len(vstec)/Nsub, scaling='spectrum', window='hamming')
                plt.semilogy(f, Pxx_den, linewidth = 0.63, label = f'{t} satellite {count+1} session', c = colors[count_index])
                count_index += 1
    # plt.legend()
    plt.ylabel('PSD', fontsize = 12)
    plt.xlabel('Частота, Гц', fontsize = 12)
    plt.xscale('log')
    # plt.legend(loc='best')
    # plt.text(time[0], vstec[0], f'{t} satellite, {count+1} session', 
    # fontsize = 6)
    if len(dict_tec) < 15:
        plt.legend(loc='upper center', bbox_to_anchor=(1, 1), fontsize = 'xx-small')
    # plt.title(key)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_tec_spectrogram_path(dict_tec):
    for key in dict_tec.keys():
        plot_tec_spectrogram(dict_tec[key], key)

# Общий спектр ТЕС (сделана синтетическая последовательность величин ТЕС через усреднение данных по всем спутникам для одного момента времени)

def plot_tec_spec_once(tec_data):
    fig, ax = plt.subplots(figsize=(16, 9))
    count_index = 0
    Nsub = 10
    for key in tec_data.keys():
        time = tec_data[key]['time'].tolist()
        vstec = tec_data[key]['vtec'].tolist()
        Nfft = len(vstec) - 1
        nperseg = len(vstec) / Nsub
        if len(time) > 100:
            sampling_freq = 1 / (60 / discret_detect(time[1], time[2]))
            f, Pxx_den = welch(vstec, fs = sampling_freq, nperseg = len(vstec)/Nsub, noverlap=nperseg/2, scaling='spectrum', nfft=nperseg, window='hamming')
            # f, Pxx_den = welch(vstec, fs = sampling_freq, noverlap=0, nperseg = len(vstec)/Nsub, scaling='spectrum', window='hamming')
            pw_fit = piecewise_regression.Fit(np.log10(f[1:]), np.log10(Pxx_den[1:]), n_breakpoints=1)
            pw_fit.plot()
            pw_fit.summary()
            # plt.show()
            # plt.semilogy(f, Pxx_den, linewidth = 0.63)
            count_index += 1
        # plt.legend()
        plt.ylabel('PSD, дБ')
        plt.xlabel('Частота, Гц')
        plt.xscale('log')
        # plt.legend(loc='best')
        # plt.title(key)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_bz(bz, DoYs, DoYe, dict_tec):
    bz = bz[(bz['day'] >= DoYs) & (bz['day'] <= DoYe) & (bz['time_decimal'] >= dict_tec[0]) & (bz['time_decimal'] <= dict_tec[-1])]
    return bz
    
# Кеограмма
 
def plot_roti_heatmap(dict_tec, window_size, lat_window, datez):
    time_mean, latitude_mean, roti, longitude_mean = [], [], [], [] 
    for t in list(dict_tec.keys()):
        for count in range(len(dict_tec[t])):
            time_surr, roti_surr, latitude_surr, longitude_surr = graph_roti(dict_tec, t, count)
            time_mean.extend(time_surr)
            latitude_mean.extend(latitude_surr)
            longitude_mean.extend(longitude_surr)
            roti.extend(roti_surr)
    
    dtime = datetime.strptime(datez[0], "%d.%m.%Y")
    mlat, mlon, mlt = geographic_to_magnetic_latitude(latitude_mean, longitude_mean, 110, dtime)    
    time_bins = np.linspace(min(time_mean), max(time_mean), num = int(window_size))
    lat_bins = np.linspace(min(mlat), max(mlat), num = lat_window)
    # H, xedges, yedges = np.histogram2d(time_mean, latitude_mean, bins = [time_bins, lat_bins], weights = roti)
    # H_avg = H.T  
    H_counts, xedges, yedges = np.histogram2d(time_mean, mlat, bins=[time_bins, lat_bins])
    H_sums, _, _ = np.histogram2d(time_mean, mlat, bins=[time_bins, lat_bins], weights=roti)
    H_avg = np.divide(H_sums, H_counts, out=np.zeros_like(H_sums), where=H_counts != 0).T  # 
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    plt.figure(figsize=(16, 9))
    imshow = plt.imshow(H_avg, extent = extent, origin = 'lower', cmap = 'jet', aspect = 'auto', interpolation = 'none')
    plt.xlabel('UTC, часы', fontsize = 20)
    plt.ylabel('MLat, градусы', fontsize = 20)
    # plt.title(f'ROTI Кеограмма с \n {((max(time_mean)-min(time_mean)) * 60) // window_size} мин в клетке {((max(mlat)-min(mlat))) // lat_window} градусов на клетку')
    cbar = plt.colorbar(orientation = 'horizontal', fraction = .1)
    cbar.set_label(label='<ROTI>, TECu/мин', fontsize = 20, labelpad = 10)
    plt.clim(.05,.25)
    plt.tight_layout()
    plt.show()

def plot_roti_heatmap1(dict_tec, bz, ae, sym, DoYs, DoYe, window_size, lat_window):
    n = 4
    l = [bz,ae,sym]
    size_plot = [5, 1, 1, 1]
    for ind in l:
        if len(ind) == 0:
            n -= 1
            size_plot.pop(-1)
            
    filt_tec = {}
    for station, data in dict_tec.items():
        filt_tec.update({str(satellite) + str(station): value for satellite, value in data.items()})
    
    time_mean, latitude_mean, roti, longitude_mean = [], [], [], [] 
    for t in list(filt_tec.keys()):
        for count in range(len(filt_tec[t])):
            time_surr, roti_surr, latitude_surr, longitude_surr = graph_roti(filt_tec, t, count)
            time_mean.extend(time_surr)
            latitude_mean.extend(latitude_surr)
            longitude_mean.extend(longitude_surr)
            roti.extend(roti_surr)
    
    # dtime = datetime.strptime(datez[0], "%d.%m.%Y")
    mlat, mlon, mlt = geographic_to_magnetic_latitude(latitude_mean, longitude_mean, 110, DoYs)    
    time_bins = np.linspace(min(time_mean), max(time_mean), num = int(window_size))
    lat_bins = np.linspace(min(mlat), max(mlat), num = lat_window)
     
    H_counts, xedges, yedges = np.histogram2d(time_mean, mlat, bins=[time_bins, lat_bins])
    H_sums, _, _ = np.histogram2d(time_mean, mlat, bins=[time_bins, lat_bins], weights=roti)
    H_avg = np.divide(H_sums, H_counts, out=np.zeros_like(H_sums), where=H_counts != 0).T  # 
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
    fig, axs = plt.subplots(n,1, layout='constrained', figsize=(16,9), gridspec_kw={'height_ratios': size_plot})
    imshow = axs[0].imshow(H_avg, extent = extent, origin = 'lower', cmap = 'jet', aspect = 'auto', interpolation = 'none')
    # axs[0].set_title('Keogram', fontsize=20)
    axs[0].set_ylabel('MLat, градусы', fontsize=15)
    axs[0].set_xticks([])
    x_min, x_max = min(time_mean), max(time_mean)


    def set_utc_ticks(ax, data, name):
        if len(data) > 0:
            time_min = np.floor(min(data['TIME_DECIMAL'])) 
            time_max = np.ceil(max(data['TIME_DECIMAL']))   
            tick_positions = np.arange(time_min, time_max + 1, 3*(1+DoYe-DoYs)) 
            tick_labels = [data[name].iloc[(data['TIME_DECIMAL'] - tick).abs().argmin()] for tick in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=12)
            # ax.set_xlim(time_min, time_max)
            ax.set_xlim(x_min, x_max)
    if len(bz) > 0:
        axs[1].plot(bz['TIME_DECIMAL'], bz['bz(GSE)'])
        set_utc_ticks(axs[1], bz, 'TIME') 
        axs[1].set_ylabel('Bz, нТ',fontsize=8) 
        axs[1].set_xticks([])
        # axs[1].set_xlim(min(time_surr), max(time_surr))  
        
    if len(ae) > 0:    
        axs[2].plot(ae['TIME_DECIMAL'], ae['AE'])
        set_utc_ticks(axs[2], ae, 'TIME_UTC') 
        axs[2].set_ylabel('AE, нТ',fontsize=8)
        axs[2].set_xticks([])
        # axs[2].set_xlim(min(time_surr), max(time_surr))

    if len(sym) > 0:
        axs[3].plot(sym['TIME_DECIMAL'], sym['SYM-H'])
        set_utc_ticks(axs[3], sym, 'TIME_UTC') 
        axs[3].set_ylabel('SYM-H, нТ',fontsize=8)
        axs[3].set_xlabel('UTC', fontsize=20)
        # axs[3].set_xlim(min(time_surr), max(time_surr))

    # axs1 = plt.gca()
    # axs1.xaxis.set_major_locator(MaxNLocator(nbins=2*(1+DoYe-DoYs)))
    cbar = fig.colorbar(axs[0].images[0], ax = axs,  orientation='vertical', fraction=0.1)
    cbar.set_label('<ROTI>, TECu/мин', fontsize=20)
    plt.show()
    
def plot_roti_heatmap_eur_am(dict_tec, bz, ae, sym, DoYs, DoYe, window_size, lat_window, ds):
    n = 5
    l = [bz,ae,sym]
    size_plot = [5, 5, 1, 1, 1]
    for ind in l:
        if len(ind) == 0:
            n -= 1
            size_plot.pop(-1)
    def roticoord(dictdata):
        filt_tec = {}
        for station, data in dictdata.items():
            filt_tec.update({str(satellite) + str(station): value for satellite, value in data.items()})
        
        time_mean, latitude_mean, roti, longitude_mean = [], [], [], [] 
        for t in list(filt_tec.keys()):
            for count in range(len(filt_tec[t])):
                time_surr, roti_surr, latitude_surr, longitude_surr = graph_roti(filt_tec, t, count)
                time_mean.extend(time_surr)
                latitude_mean.extend(latitude_surr)
                longitude_mean.extend(longitude_surr)
                roti.extend(roti_surr)
        
        # dtime = datetime.strptime(datez[0], "%d.%m.%Y")
        mlat, mlon, mlt = geographic_to_magnetic_latitude(latitude_mean, longitude_mean, 110, ds)   
        time_bins = np.linspace(min(time_mean), max(time_mean), num = int(window_size))
        lat_bins = np.linspace(min(mlat), max(mlat), num = lat_window)
        
        H_counts, xedges, yedges = np.histogram2d(time_mean, mlat, bins=[time_bins, lat_bins])
        H_sums, _, _ = np.histogram2d(time_mean, mlat, bins=[time_bins, lat_bins], weights=roti)
        H_avg = np.divide(H_sums, H_counts, out=np.zeros_like(H_sums), where=H_counts != 0).T  # 
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        x_min, x_max = min(time_mean), max(time_mean)
        return H_avg, extent, x_min,x_max
    
    
    stations_na = ['INEG', 'MDO1', 'DUBO', 'CHUR', 'BAKE', 'EUR', 'NLIB', 'PIE1', 'INVK', 'YELL', 'GJOC', 'GILC', 'RESC', 'RESO', 'SSIA', 'SASK', 'FLIN', 'P053', 'P051', 'PONC', 'TALC',
                   'MAWY', 'TSWY', 'BLW2', 'P032', 'P012', 'NMRO']
    stations_eur = ['KIRU','MET', 'RIGA', 'LAMA', 'GANP', 'BUCU', 'ORID', 'NOT' , 'WUTH', 'METS', 'NOT1', 'MET3', 'NYA', 'NYA1', 'GLSV', 'VIS', 'MAR', 'TRO', 'NAGB',
                    'VARS', 'JOEN', 'KIV', 'OUL', 'KUU', 'SODA', 'KEV', 'HONS', 'ESOX', 'TEJH', 'SUN6', 'BJU', 'MAR6']
    # stations_eur = ['KIRU','SVTL', 'RIGA', 'LAMA', 'BUCU','POLV', 'ORID', 'NICO', 'WUTH']
    print(dict_tec.keys())
    filt_eur, filt_na = {}, {}
    # for station in stations_eur:
    #     if station in dict_tec:
    #         filt_eur[station] = dict_tec[station]
    # for station in stations_na:
    #     if station in dict_tec:
    #         filt_na[station] = dict_tec[station]
    for station in stations_eur:
        for key in dict_tec.keys():
            if station.startswith(key):
                filt_eur.update({station: dict_tec[key]})
    for station in stations_na:
        for key in dict_tec.keys():
            if station.startswith(key):
                filt_na.update({station: dict_tec[key]})
    
    
    # print(filt_eur.keys())
    # print(filt_na.keys())
    # print(filt_eur)
    H_avg_eur, extent_eur, x_min_eur, x_max_eur = roticoord(filt_eur)
    H_avg_na, extent_na, x_min_na, x_max_na = roticoord(filt_na)
    
    fig, axs = plt.subplots(n,1, layout='constrained', figsize=(16,9), gridspec_kw={'height_ratios': size_plot})
    imshow1 = axs[0].imshow(H_avg_eur, extent = extent_eur, origin = 'lower', cmap = 'jet', aspect = 'auto', interpolation = 'none', vmin = 0.05, vmax = 0.25)
    imshow2 = axs[1].imshow(H_avg_na, extent = extent_na, origin = 'lower', cmap = 'jet', aspect = 'auto', interpolation = 'none', vmin = 0.05, vmax = 0.25)
    axs[0].set_title('Европейский регион', fontsize=10)
    axs[1].set_title('Североамериканский регион', fontsize=10)
    axs[0].set_ylabel('MLat, градусы', fontsize=15)
    axs[0].set_xticks([])
    axs[1].set_ylabel('MLat, градусы', fontsize=15)
    axs[1].set_xticks([])
    
    def set_utc_ticks(ax, data, name):
        if len(data) > 0:
            time_min = np.floor(min(data['TIME_DECIMAL']))  
            time_max = np.ceil(max(data['TIME_DECIMAL']))   
            tick_positions = np.arange(time_min, time_max + 1, 3*(1+DoYe-DoYs))  
            tick_labels = [data[name].iloc[(data['TIME_DECIMAL'] - tick).abs().argmin()] for tick in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=12)
            ax.set_xlim(x_min_eur, x_max_eur)
        
    if len(bz) > 0:
        axs[2].plot(bz['TIME_DECIMAL'], bz['bz(GSE)'])
        set_utc_ticks(axs[2], bz, 'TIME')  
        axs[2].set_ylabel('Bz, нТ',fontsize=8)
        axs[2].set_xticks([])
        # axs[2].set_xlim(min(time_surr), max(time_surr))  
        
    if len(ae) > 0:    
        axs[3].plot(ae['TIME_DECIMAL'], ae['AE'])
        set_utc_ticks(axs[3], ae, 'TIME_UTC') 
        axs[3].set_ylabel('AE, нТ',fontsize=8)
        axs[3].set_xticks([])
        # axs[3].set_xlim(min(time_surr), max(time_surr))

    if len(sym) > 0:
        axs[4].plot(sym['TIME_DECIMAL'], sym['SYM-H'])
        set_utc_ticks(axs[4], sym, 'TIME_UTC') 
        axs[4].set_ylabel('SYM-H, нТ',fontsize=8)
        axs[4].set_xlabel('UTC', fontsize=20)
        # axs[4].set_xlim(min(time_surr), max(time_surr))
    
    # axs1 = plt.gca()
    # axs1.xaxis.set_major_locator(MaxNLocator(nbins=2*(1+DoYe-DoYs)))
    cbar = fig.colorbar(axs[0].images[0], ax = axs,  orientation='vertical', fraction=0.1)
    cbar.set_label('<ROTI>, TECu/мин', fontsize=20, labelpad = 5)
    # plt.clim(.05,.25)
    plt.show()    
    
def plot_roti_heatmap_path(dict_tec):
    filt_tec = {}
    for station, data in dict_tec.items():
        filt_tec.update({str(satellite) + str(station): value for satellite, value in data.items()})
    plot_roti_heatmap(filt_tec)

# Построение вейвлет-спектрограммы
 
def plot_wavelet(dict_tec, filter_name, flaglist):
    time_surr, tec_surr = [],[]
    for t in list(dict_tec.keys()):
        for count in range(len(dict_tec[t])):
            time, tec = graph_tec(dict_tec, t, count)
            time_surr.extend(time)
            tec_surr.extend(tec)
                   
    data = pd.DataFrame({'time': time_surr, 'tec': tec_surr})
    mean_value = data.groupby('time')['tec'].mean().reset_index()
    data = data.drop(columns = ['tec']).drop_duplicates(subset=['time']).merge(mean_value, on ='time')  
    data = data.sort_values(by='time').reset_index(drop=True)
    if ((filter_name != 'decimation') and (flaglist[4])):   
        data = decimation_2(data).reset_index(drop=True) 
    
    # scales = np.geomspace(1, 1024, num=100)
    # sampling_freq = 1 / (60 / discret_detect(data['time'].tolist()[2], data['time'].tolist()[3]))
    # coeff, freq = pywt.cwt(data['tec'], scales, wavelet = 'cmor0.5-1.0', sampling_period=sampling_freq)
    # coeff = np.abs(coeff[:-1, :-1])
    # period = 1. / freq
    # plt.figure()
    
    scales = scg.periods2scales(np.arange(1/15,3,0.1))
    scg.cws(time = np.array(data['time'].tolist()),signal = data['tec'], scales=np.arange(50,256,0.1), coikw=coikw, cmap="jet", cbarlabel='PSD', ylabel='Период', xlabel="Время", yscale="log", title='CWT', wavelet = 'cmor3-1.5', spectrum = 'power') 
    
    # смотреть тут https://github.com/alsauve/scaleogram/blob/master/lib/scaleogram/cws.py
    # spectrum = 'power' for abs(CWT)**2
    # plt.imshow(coeff, extent=[data['time'].tolist()[0], data['time'].tolist()[-1], period[-1], period[0]], aspect='auto',
    #         interpolation='bicubic', cmap='viridis')
    # plt.colorbar(label='Абс. Величина')
    # plt.ylabel('Period')
    # plt.xlabel('Time (s)')
    # plt.yscale('log')
    # plt.ylim(0, max(freq))
    plt.title('Скалограмма')
    plt.show()

def plot_wavelet_path(dict_tec, filter_name, flaglist):
    for key in dict_tec.keys():
        plot_wavelet(dict_tec[key], filter_name, flaglist)

# Спектрограмма ТЕС с графиками геомагнитных индексов
 
def plot_sfft_spec(dict_tec, bz, ae, sym, window_size, DoYs, DoYe, filter_name, flaglist):
    n = 4
    l = [bz,ae,sym]
    size_plot = [5, 1, 1, 1]
    for ind in l:
        if len(ind) == 0:
            n -= 1
            size_plot.pop(-1)
    time_surr, tec_surr = [],[]
    for t in list(dict_tec.keys()):
        for count in range(len(dict_tec[t])):
            time, tec = graph_tec(dict_tec, t, count)
            time_surr.extend(time)
            tec_surr.extend(tec)
                   
    data = pd.DataFrame({'time': time_surr, 'tec': tec_surr})
    mean_value = data.groupby('time')['tec'].mean().reset_index()
    data = data.drop(columns = ['tec']).drop_duplicates(subset=['time']).merge(mean_value, on ='time')  
    data = data.sort_values(by='time').reset_index(drop=True)
    if ((filter_name != 'decimation') and (flaglist[4])):
        data = decimation_2(data) 
    window = time_lag(min(data['time']), max(data['time'])) 
    sampling_freq = 1 / (60 / discret_detect(data['time'][1], data['time'][2]))
    
    fig, axs = plt.subplots(n,1, layout='constrained', figsize=(16,9), gridspec_kw={'height_ratios': size_plot})
    NFFT = int(round(len(data['tec'].tolist()) // ((window * 60)) * window_size)) 
    Sxx, frequencies, bins, im = axs[0].specgram(data['tec'], Fs=sampling_freq, mode = 'psd', NFFT = NFFT, detrend = 'linear', noverlap = NFFT // 2, cmap = 'inferno')
    axs[0].set_title('SFFT Спектрограмма', fontsize=12)
    axs[0].set_ylabel('Частота, Гц', fontsize=8)
    # ax1.set_yscale('log')
    axs[0].set_ylim(0, max(frequencies))
    # num_ticks = 10 
    # tick_positions = bins[::len(bins)//num_ticks]  
    # tick_labels = [f"{int(np.floor(x))}:{str(round(x % 1 * 60)).zfill(2)}" for x in data['time'].tolist()[::len(data['time'].tolist()) // num_ticks]]
    # if len(tick_positions) != len(tick_labels):
    #     min_length = min(len(tick_positions), len(tick_labels))
    #     tick_positions = tick_positions[:min_length]
    #     tick_labels = tick_labels[:min_length]
    # axs[0].set_xticks(ticks=tick_positions, labels=tick_labels)
    axs[0].set_xticks([])
    
    if len(bz) > 0:
        bz = plot_bz(bz, DoYs, DoYe, data['time'].tolist())
        axs[1].plot(bz['time'], bz['bz(GSE)'])
        axs[1].set_ylabel('Bz, нТ',fontsize=8)
        # axs[1].set_xlabel('Время', fontsize=8)
        axs[1].set_xticks([])
        axs[1].set_xlim(min(time_surr), max(time_surr))
    
    if len(ae) > 0:    
        ae = ae[(ae['TIME_DECIMAL'] >= data['time'].tolist()[0]) & (ae['TIME_DECIMAL'] <= data['time'].tolist()[-1])]
        axs[2].plot(ae['TIME_UTC'], ae['AE'])
        axs[2].set_ylabel('AE, нT',fontsize=8)
        # axs[2].set_xlabel('Время', fontsize=8)
        axs[2].set_xticks([])
        axs[2].set_xlim(min(time_surr), max(time_surr))
    
    if len(sym) > 0:
        sym = sym[(sym['TIME_DECIMAL'] >= data['time'].tolist()[0]) & (sym['TIME_DECIMAL'] <= data['time'].tolist()[-1])]
        axs[3].plot(sym['TIME_UTC'], sym['SYM-H'])
        axs[3].set_ylabel('SYM-H, нT',fontsize=8)
        axs[3].set_xlabel('UTC, часы', fontsize=8)
        # axs[3].set_xticks([])
        axs[3].set_xlim(min(time_surr), max(time_surr))
    
    axs1 = plt.gca()
    axs1.xaxis.set_major_locator(MaxNLocator(nbins=10))
    cbar = fig.colorbar(axs[0].images[0], ax = axs,  orientation='vertical', fraction=0.1)
    # cbar = fig.colorbar(ax1.images[0], ax = (ax1, ax2),  location = 'bottom', fraction=0.07)
    cbar.set_label('Спектральная плотность мощности')
    # lmin = np.min(Sxx)
    # lmax = np.max(Sxx)
    # num_ticks = 5
    # ticks = np.linspace(lmin, lmax, num_ticks)
    # cbar.set_ticks(ticks)
    # cbar.minorticks_on()
    plt.show()
    
def plot_sfft_spec_path(dict_tec, filter_name, flaglist):
    for key in dict_tec.keys():
        plot_sfft_spec(dict_tec[key], filter_name, flaglist)
        
# Построение карты с поточечным ROTI (где указывается координата подыоносферной точки)
 
def current_roti(narrow_dict, satellite, count, vremya):
    # utc = datetime.strptime(vremya, '%H:%M:%S')
    roti_time = datetime2decimals(vremya)
    rot, time, latitude = [], [], []
    rot = np.abs(narrow_dict['vtec'].diff().iloc[1:]) 
    time = np.mean(narrow_dict['time']) 
    latitude = np.mean(narrow_dict['lat'])
    longitude = np.mean(narrow_dict['lon'])
    roti = np.std(rot)
    return [roti, longitude, latitude, time]
    
def current_roti_scatter(vremya, dict_tec):
    roti_time = datetime2decimals(vremya)
    roti_surr, lon_surr, lat_surr, time_surr = [],[],[],[]
    for satellite in list(dict_tec.keys()):
        for count in range(len(dict_tec[satellite])):
            dict_tec1 = pd.DataFrame(copy.copy(dict_tec[satellite][count]), columns=['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
            dict_tec1 = dict_tec1.drop_duplicates(subset=['time']).sort_values(by='time').reset_index(drop=True)
            # fs = 1 / (np.mean(np.diff(dict_tec1['time'])) * 3600)
            try:
                fs = 1 / (60 / discret_detect(dict_tec1['time'].tolist()[1], dict_tec1['time'].tolist()[2]))
            except ZeroDivisionError:
                continue
                
            dict_tec1 = dict_tec1.iloc[::int(round(60 * fs))].reset_index(drop=True)
            narrow_dict = dict_tec1[(dict_tec1['time'] >= roti_time - (1/60*3)) & (dict_tec1['time'] <= roti_time + (1/60*3))]
            if narrow_dict[['vtec', 'time', 'lat', 'lon']].isnull().values.any():
                continue
            roti, longitude, latitude, time = current_roti(narrow_dict, satellite, count, vremya)
            if not np.isnan(roti):
                time_surr.append(time)
                lon_surr.append(longitude)
                lat_surr.append(latitude)
                roti_surr.append(roti)
    
    return [time_surr, lon_surr, lat_surr, roti_surr]

def plot_current_roti(utc, dict_tec, kp, dates, flux):
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', linewidth = .3, facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='face', linewidth = .3, facecolor=cfeature.COLORS['water'])
    utc_datetime = datetime.strptime(utc, '%H:%M:%S')
    time_surr, lon_surr, lat_surr, roti_surr = current_roti_scatter(utc_datetime, dict_tec)
    geo_dots = 2400
    lambds = np.linspace(0, 360, geo_dots)
    eps = np.linspace(30, 90, geo_dots) 
    utc = datetime.strptime(utc, '%H:%M:%S').time()  
    dtime = datetime.combine(dates, utc)
    
    [mlat, mlon, mlt] = geographic_to_magnetic_latitude(eps, lambds, 110, dtime)
    mlt = convert_time_to_mlt(dtime, mlon)
    [mlat_roti, mlon_roti, _] = geographic_to_magnetic_latitude(lat_surr, lon_surr, 110, dtime)
    # mlon_roti -= 90
    (q, elat, plat) = ZP_oval(mlt, kp, mlat, flux)

    # orthograph = ccrs.Orthographic(central_lon, central_lat)
    # orthograph._threshold /= 10

    plt.figure()
    # ax = plt.axes(projection=orthograph)
    ax = plt.axes(projection = ccrs.PlateCarree())
    ax.set_global()
    # ax.set_extent(extent)
    ax.gridlines(color ='black', linewidth = .1)
    ax.add_feature(ocean, zorder = 1)
    # ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(land)
    # ax.add_feature(cfeature.nightshade.Nightshade(time, alpha=0.2))
    ax.coastlines(resolution = '50m', color =  'black', linewidth = .01)
    # for location in [saintp, svtl, mdvj, zeck, polv, kiru, riga, lama, uzhl, bucu]:
    #         ax.plot(location['lat'], location['lon'], 'rx', markersize=3.5, zorder = 3,transform=ccrs.Geodetic())

    ax.plot(mlon, plat, 'g', linewidth=1.5, zorder = 3, transform=ccrs.Geodetic())
    ax.plot(mlon, elat, 'g', linewidth=1.5, zorder = 4, transform=ccrs.Geodetic())
    roti_surr_array = np.array(roti_surr)
    alpha = np.where(roti_surr_array < 0.02, 0, 0.9)
    scatter = ax.scatter(lon_surr, mlat_roti, s = roti_surr_array * 50, c = roti_surr_array, cmap = 'jet', marker = 'o',norm = colors.Normalize(vmin = 0.05, vmax = 0.25), alpha = alpha, transform = ccrs.PlateCarree(), zorder = 7)
    plt.title(f'{dates.date()} {utc} UTC')
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', fraction=0.06, pad=0.02)
    cbar.set_label('ROTI, TECu/мин', labelpad=5)
    plt.tight_layout()
    plt.show()
        
def plot_current_roti_path(utc_time, dict_tec, kp, dates, *args):
    if args:
        flux = args
    else:
        flux = 0.2
    filt_tec = {}
    for station, data in dict_tec.items():
        filt_tec.update({str(satellite) + str(station): value for satellite, value in data.items()})
    plot_current_roti(utc_time, filt_tec, kp, dates, flux)


def MI_plot(processed_data, ae, sym, bz):  
      
    roti_dfs = [] 

    for sat in processed_data.keys():
        for count in range(len(processed_data[sat])):
            [time_mean, roti_values, _, _] = graph_roti(processed_data, sat, count)
            temp_roti_df = pd.DataFrame({'TIME_DECIMAL': time_mean, 'roti': roti_values})
            roti_dfs.append(temp_roti_df)
            roti = pd.concat(roti_dfs, ignore_index=True) 

    # roti = (roti
    #             .groupby('TIME_DECIMAL', as_index=False)  # Keep as DataFrame
    #             .agg(roti_mean=('roti', 'mean'))  # Calculate mean per time
    #             .sort_values('TIME_DECIMAL')
    #             .reset_index(drop=True)
    #             ) 
    # roti = roti.rename(columns = {'TIME_DECIMAL': 'TIME_DECIMAL', 'roti_mean':'roti'}) 

    keys_to_remove = []
    geo_dict = {'AE': ae,
                'SYM-H': sym,
                'bz(GSE)': bz}

    for key, value in geo_dict.items():
        if len(value) == 0:  
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del geo_dict[key]

    # print(geo_dict)
    images = []
    fig, axs = plt.subplots(1, len(geo_dict), sharex = False, sharey = False, figsize = (15,5))
    for i, (key, value) in enumerate(geo_dict.items()):
        
        # value = value[(value['TIME_DECIMAL'] <= max(roti['TIME_DECIMAL'].tolist())) & (value['TIME_DECIMAL'] >= min(roti['TIME_DECIMAL'].tolist()))].reset_index(drop=True)
        # both = pd.merge_asof(roti, value, on='TIME_DECIMAL', direction='nearest', tolerance=.01667).dropna() # если надо не усредненный за минуту ROTI,
        # а присвоенный каждому ROTI геомаг. индекс в эту минуту
        
        # если нужен усреденный за минуту ROTI
        minute_step = 1 / 60  # 0.01666...
        roti['TIME_DECIMAL_MIN'] = (roti['TIME_DECIMAL'] // minute_step) * minute_step
        value['TIME_DECIMAL_MIN'] = (value['TIME_DECIMAL'] // minute_step) * minute_step
        value = value[(value['TIME_DECIMAL_MIN'] <= max(roti['TIME_DECIMAL_MIN'].tolist())) & (value['TIME_DECIMAL_MIN'] >= min(roti['TIME_DECIMAL_MIN'].tolist()))].reset_index(drop=True)
        roti_mean_per_minute = roti.groupby('TIME_DECIMAL_MIN', as_index=False).agg({'roti': 'mean'}) 
        both = pd.merge(roti_mean_per_minute, value, on='TIME_DECIMAL_MIN', how='inner').dropna() 

        hist_roti, bins_roti = np.histogram(both['roti'], density = True)
        hist_ae, bins_ae = np.histogram(both[key], density = True)
        heatmap, xedges, yedges = np.histogram2d(both['roti'], both[key], density = True, range =[[bins_roti[0], bins_roti[-1]],[bins_ae[0], bins_ae[-1]]])
        bins_roti, bins_ae, xedges, yedges = list(map(extend_edges, [bins_roti, bins_ae, xedges, yedges]))
        both['roti_prob'] = both['roti'].apply(lambda x: probability_from_histogram(x, hist_roti, bins_roti))
        both['ae_prob'] = both[key].apply(lambda x: probability_from_histogram(x, hist_ae, bins_ae))
        both['joint_prob'] = both.apply(lambda row: joint_prob(row, heatmap, xedges, yedges, key), axis = 1)
        # both['pmi'] = both.apply(lambda row: pmi(row), axis = 1)
        both['pmi'] = both.apply(lambda row: norm_pmi(row), axis = 1)
        # both.to_csv(r'd:\data\output.csv')
        # print(both)
        heatmap2, _, _ = np.histogram2d(both['roti'], both[key], weights=both['pmi'], bins = (xedges, yedges))
        heatmap_count, xedges,yedges = np.histogram2d(both['roti'], both[key], bins = (xedges, yedges))
        H_avg = np.divide(heatmap2, heatmap_count, out=np.zeros_like(heatmap2), where=heatmap_count != 0).T  # 
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        image = axs[i].imshow(H_avg, extent=extent, origin='lower', aspect='auto', interpolation='nearest', cmap='viridis', vmin = -2, vmax = 2)
        images.append(image)

        axs[i].set_xlabel('ROTI, TECu/min')
        axs[i].set_ylabel(f'{key}, nT')
        axs[i].set_title(f'ROTI and {key} with PMI as weights', size = 10)
        ymin = value[key].min()
        ymax = value[key].max()
        axs[i].set_ylim(ymin,ymax)
        

    for i, ax in enumerate(axs):
        fig.colorbar(images[i], ax=ax, fraction=0.08, pad=0.04)

    fig.subplots_adjust(wspace = 5)
    plt.tight_layout()
    plt.show()