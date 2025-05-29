import numpy as np
import copy
from datetime import datetime
import pandas as pd
from scipy.signal import butter, filtfilt, decimate, lfilter, welch, firwin, cheb1ord, cheby1, remez, sosfilt
import aacgmv2
from scipy.io import loadmat

time_per_degree = 360/(24*3600)

def discret_detect(t1,t2):
    return round(1 / (time_lag(t1,t2) * 60))

def rolling_mean(data, window=20):
    return np.mean(data[-window:])

def convert_time_to_local(time, longitude):
    
    utc_time = datetime.strptime(time, '%H:%M:%S')
    time = datetime2decimals(utc_time)
    
    if longitude <= 0:
        time -= (abs(longitude) / time_per_degree) * (1/3600)
        if time < 0:
            time += 24.0
    else:
        time += (longitude / time_per_degree) * (1/3600)

    return time 

def convert_time_to_mlt(standard_time, longitude):

    magnetic_longitude = longitude
    
    mlt = aacgmv2.convert_mlt(magnetic_longitude, standard_time)
    return mlt

def time_lag(t_i1, t_i2):
    time_diff = 0
    if abs(t_i1) > abs(t_i2):
        time_diff = abs(t_i2) + 24.0 - abs(t_i1)
    else:
        time_diff = t_i2 - t_i1
    return np.round((np.floor(time_diff) + (time_diff % 1)), 5)

def datetime2decimals(time):
    hours = time.hour
    minutes = time.minute
    seconds = time.second
    decimal_time = hours + (minutes / 60) + (seconds / 3600)
    
    return decimal_time

def butter_lowpass(sampling_freq, wn, order = 5):
    nyq = 0.5 * wn
    # normal_cutoff = cutoff / nyq
    b, a = butter(order, nyq, btype='low', analog=False, fs = sampling_freq)
    return b, a

def lowpass_filter(data, sampling_freq, wn, order = 5):
    b, a = butter_lowpass(sampling_freq, wn, order = order)
    y = filtfilt(b, a, data)
    return y

def butterfilt(dict_tec, wn):
    dict_tec = pd.DataFrame(dict_tec, columns=['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
    sampling_freq = 1 / (60 / discret_detect(dict_tec['time'][2], dict_tec['time'][3]))
    if sampling_freq > wn / 2:
        vtec= lowpass_filter(dict_tec['vtec'].values, sampling_freq, wn)
        dict_tec['vtec'] = vtec
    else:
        pass
    return dict_tec.values.tolist()

def cheb(dict_tec, wn):
    dict_tec = pd.DataFrame(dict_tec, columns=['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
    sampling_freq = 1 / (60 / discret_detect(dict_tec['time'][2], dict_tec['time'][3]))
    if sampling_freq > wn / 2:
        N, W_n = cheb1ord(wn, (wn**-1 - 5)**-1, 0.5, 40, fs = sampling_freq)
        sos = cheby1(N, 5, W_n, fs = sampling_freq, output = 'sos')
        dict_tec['vtec'] = sosfilt(sos, dict_tec['vtec'])
        dict_tec = dict_tec.iloc[N+50:].reset_index(drop=True)
    else:
        pass
    return dict_tec.values.tolist()

def remezfilt(dict_tec, wn):
    dict_tec = pd.DataFrame(dict_tec, columns=['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
    tapz = 128
    sampling_freq = 1 / (60 / discret_detect(dict_tec['time'][2], dict_tec['time'][3]))
    if sampling_freq > wn / 2:
        fir_coeffs = remez(numtaps = tapz, bands = [0, wn, (wn**-1 - 3)**(-1), 0.5 * sampling_freq], desired = [1, 0], fs = sampling_freq) 
        dict_tec['vtec'] = lfilter(fir_coeffs, [1], dict_tec['vtec'])
        # dict_tec['vtec'] = sosfilt(taps, dict_tec['vtec'])
        dict_tec = dict_tec.iloc[tapz:].reset_index(drop=True)
    else:
        pass
    return dict_tec.values.tolist()

def firwinfilt(dict_tec, wn):
    dict_tec = pd.DataFrame(dict_tec, columns=['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
    sampling_freq = 1 / (60 / discret_detect(dict_tec['time'][2], dict_tec['time'][3]))
    if sampling_freq > wn / 2:
        taps = 128
        h = firwin(taps, wn, window=('kaiser', 9), pass_zero = True)
        vtec= lfilter(h, 1, dict_tec['vtec'].values)
        dict_tec['vtec'] = vtec
        dict_tec = dict_tec.iloc[taps:].reset_index(drop=True)
    else:
        pass
    return dict_tec.values.tolist()

def decimation(dict_tec, wn):
    dict_tec = pd.DataFrame(dict_tec, columns=['time', 'satellite', 'az', 'el', 'lat', 'lon', 'stec', 'vtec'])
    sampling_freq = 1 / (60 / discret_detect(dict_tec['time'][2], dict_tec['time'][3]))
    if sampling_freq > wn / 2:
        decimation_factor = int(np.ceil(sampling_freq * 30))
        while decimation_factor > 1:
            if decimation_factor > 13:
                vtec = decimate(dict_tec['vtec'].values, q = 10, ftype = 'fir', n = 4)
                dict_tec = dict_tec.iloc[::10]
                dict_tec['vtec'] = vtec        
                decimation_factor = decimation_factor // 10 
            else:
                vtec = decimate(dict_tec['vtec'].values, q = decimation_factor, ftype = 'iir', n = 8)
                dict_tec = dict_tec.iloc[::decimation_factor]
                dict_tec['vtec'] = vtec
                decimation_factor -= decimation_factor
            dict_tec = dict_tec.iloc[10:].reset_index(drop=True)    
    else:
        pass
    return dict_tec.values.tolist()

def decimation_2(data):
    sampling_freq = 60 / discret_detect(data['time'].tolist()[0], data['time'].tolist()[1])
    data = data.iloc[::int(30/sampling_freq)].reset_index(drop=True)  
    return data 

def phi_t(time):
    return (3.16 - 5.6 * np.cos(np.radians(15.0 *(time - 2.4))) + 1.4 * np.cos(np.radians(15.0 * (2 * time - 0.8))))
def phi_N(lamb):
    return (0.85 * np.cos(np.radians(lamb + 63.0)) - 0.52 * np.cos(np.radians(2.0 * lamb + 5.0)))
def phi_min_boundary(kp_m, time, lamb):
    return (65.5 - 2.4 * kp_m + phi_t(time) + phi_N(lamb) * np.exp(-0.3 * kp_m) )
def phi_min_polar_boundary(time, k):
    return (76.0 + 0.3 * k - (1 - 0.1 * k) * np.cos(np.radians(15.0 * time)))
def phi_avr(time, k):
    return (74.0) - 5.0 * np.cos(np.radians(15 * time)) - 1.4 * k

def geographic_to_magnetic_latitude(geo_latitude, geo_longitude, altitude, dtime):
    [magnetic_latitude, magnetic_longitude, mlt] = aacgmv2.get_aacgm_coord_arr(geo_latitude, geo_longitude, altitude, dtime, method = 'G2A')
    return magnetic_latitude, magnetic_longitude, mlt

def calc_HP(kp):
    if kp<=5:
         HP = 38.66*np.exp(0.1967*kp)-33.99
    elif kp>5:
        HP = 4.592*np.exp(0.4731*kp)+20.47
    return HP

def oval_bound(mlt, kp, pelim, mlat, Q):
      cm = np.where(Q>=pelim, 1, 0) # comparison matrix
      ej = np.argmax(cm, 0)
      pj = np.argmax(np.flip(cm, 0), 0)
      eqlat = mlat[ej]
      pollat = np.flip(mlat)[pj]
      k1 = np.argwhere(ej == 0)[:,0]
      N = 0
      for i in k1:
          if np.all(cm[:,i])==0:
              eqlat[i] = np.nan
              pollat[i] = np.nan
              N = N+1
      return (eqlat, pollat)

def ZP_oval(mlt, kp, mlat, *arg):
    if arg:
        pelim = arg
    else:
        pelim = 0.25 
    zfile_new = r"d:\Mine\Код\auroraloval-main\data\zhang_new.mat"
    ds = loadmat(zfile_new)
    data = ds['zhang']
    #
    kpmodel = np.array([0.75, 2.25, 3.75, 5.25, 7.00, 9.00])
    angle = mlt*2*np.pi/24
    angle0 = angle.copy().reshape(mlt.size, 1)
    angle1 = np.tile(angle0, (1, 4))
    x = 90-mlat
    x0 = x.reshape(x.size, 1)
    x1 = np.tile(x0, (1, mlt.size))
    inds = np.argwhere(np.isnan(data[:,0]))[:,0]
    #
    Qmodel = np.zeros((6, mlat.size, mlt.size))
    for m in range(6):
        i = inds[m]+1    
        b0 = data[i,:]
        bc = data[i+1:i+7,:]
        bs = data[i+7:i+13,:]
        #
        S = np.zeros((mlt.size, 4))
        for n in range(6):
            k = n+1
            b1 = bc[n,:]
            b2 = bs[n,:]
            S = S+b1*np.cos(k*angle1)+b2*np.sin(k*angle1)
        #
        A = b0+S
        A0 = A[:, 0]
        A1 = A[:, 1]
        A2 = A[:, 2]
        A3 = A[:, 3]
        #
        Qmodel[m,:,:] = (A0*np.exp((x1-A1)/A2))/(1+np.exp((x1-A1)/A3))**2
    #
    kd = kpmodel-kp
    inds = np.where(kd<0, -1, 1)
    j = np.argwhere(kd>=0)[0,0]
    if j==0:
        km = 0.75
        km1 = 2.25
        Qm = Qmodel[0,:,:]
        Qm1 = Qmodel[1,:,:]
    else:
        km = kpmodel[j-1]
        km1 = kpmodel[j]
        Qm = Qmodel[j-1,:,:]
        Qm1 = Qmodel[j,:,:]
    #
    HPm1 = calc_HP(km1)
    HPm = calc_HP(km)
    HP = calc_HP(kp)
    #
    fm = (HPm1-HP)/(HPm1-HPm)
    fm1 = (HP-HPm)/(HPm1-HPm)
    #
    Q = fm*Qm + fm1*Qm1
    (elat, plat) = oval_bound(mlt, kp, pelim, mlat, Q)
    return (Q, elat, plat)

def probability_from_histogram(value, hist, bins):
    flag =False
    bin_index = np.digitize(value, bins, right=flag) - 1
    if 0 <= bin_index < len(hist):
        bin_width = bins[bin_index+1] - bins[bin_index]  
        probability = hist[bin_index]
        return probability
    else:
        bin_width = bins[bin_index+2] - bins[bin_index-1]  
        return hist[bin_index + 1] 
    
def joint_prob(row, heatmap, xedges, yedges, key):
    flag = False
    roti = row['roti']
    ae = row[key]
    roti_index = np.digitize(roti, xedges, right=flag) - 1
    ae_index = np.digitize(ae, yedges, right =flag) - 1
    if (0 <= roti_index <= heatmap.shape[0]) and (0 <= ae_index <= heatmap.shape[1]):
        bin_area = (xedges[roti_index+1] - xedges[roti_index]) * (yedges[ae_index+1] - yedges[ae_index])
        return heatmap[roti_index, ae_index] 
    else:
        bin_area = (xedges[roti_index+2] - xedges[roti_index+1]) * (yedges[ae_index+2] - yedges[ae_index+1])
        return heatmap[roti_index+1, ae_index+1] 
        # return 0
 
def pmi(row):
    rprob = row['roti_prob']
    aprob = row['ae_prob']
    jprob = row['joint_prob']
    return  np.log2(jprob/(rprob*aprob))

def norm_pmi(row):
    rprob = row['roti_prob']
    aprob = row['ae_prob']
    jprob = row['joint_prob']
    return  np.log2(jprob/(rprob*aprob))/ (-np.log2(jprob))

def extend_edges(bin):
    eps = 1e-2
    bin[0], bin[-1] = bin[0] - eps, bin[-1] + eps 
    return bin

def addtime(index):
    # for i in range(DoYe-DoYs):
    index['delay'] = index['DOY'] - index['DOY'].min()
    index['TIME_DECIMAL'] = index.apply(lambda x: x['TIME_DECIMAL'] if x['delay'] == 0 else x['TIME_DECIMAL'] + 24 * x['delay'], axis = 1)
    
    return index
