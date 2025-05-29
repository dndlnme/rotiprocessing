from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import sys
import PySimpleGUI as sg
import os
from main import main
from utils import ftp_downloader

matplotlib.use('QtAgg')
plt.style.use('seaborn-v0_8-paper')
# plt.style.use('seaborn-v0_8-poster')
# plt.style.use('fast')
# project_root = os.path.dirname(os.path.abspath(__file__))
# stations_path = os.path.join(project_root, "stations.txt")
# stations_epmcb_path = os.path.join(project_root, "stations_epmcb.txt")
# stations_cors_chain_path = os.path.join(project_root, 'stations_cors_chain.txt')
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

stations_path = resource_path("stations.txt")
stations_cors_chain_path = resource_path("stations_cors_chain.txt")
stations_epmcb_path = resource_path("stations_epmcb.txt")

with open(stations_path, 'r') as stations:
    stations = stations.read().splitlines()
with open(stations_epmcb_path, 'r') as stations_epmcb:
    stations_epmcb = stations_epmcb.read().splitlines()
stations.extend(stations_epmcb)
with open(stations_cors_chain_path, 'r') as stations_cors_chain:
    stations_cors_chain = stations_cors_chain.read().splitlines()
stations.extend(stations_cors_chain)

filters = ['none','butter', 'cheby', 'firwin', 'decimation', 'remez']
# app = QtWidgets.QApplication([])
sg.theme('Dark Grey 10')
layout1 = [[sg.Text('Data Folder'), sg.InputText(key = '-DIRECTORY1-', enable_events=True), sg.FileBrowse()],
        [sg.Text('Bz Data Folder'), sg.InputText(key = '-DIRECTORY3-', enable_events=True), sg.FileBrowse()],
        [sg.Text('AE Data Folder'), sg.InputText(key = '-DIRECTORY4-', enable_events=True), sg.FileBrowse()],
        [sg.Text('SYM/ASY Data Folder'), sg.InputText(key = '-DIRECTORY5-', enable_events=True), sg.FileBrowse()],
        [sg.Text('Output Folder'), sg.InputText(key = '-DIRECTORY6-', enable_events=True), sg.FileBrowse()],
        [sg.Text('DoYstart, DoYend (example: 01.01.2001)', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN0-')],
        [sg.Text('Time for point ROTI/TEC', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN6-', default_text = '20:00:00')],
        [sg.Text('TimeStart (example: 15.1)', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN1-', default_text = '0')],
        [sg.Text('TimeStop', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN2-', default_text = '24')],
        [sg.Text('Elevation Angle Mask', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN3-',default_text = '30')],
        [sg.Text('Satellite Number (0 if no need)', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN4-', default_text = '0')],
        [sg.Text('Window size (mins)', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN5-', default_text = '10')],
        [sg.Text('Window size (lat)', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN7-',default_text = '30')],
        [sg.Text('Kp-index', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN8-',default_text = '8')],
        [sg.Text('Wn for filtering (1/30 = 0.0333 or 1/60 = 0.0166)', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN9-',default_text = '0.0333')],
        [sg.Text('Energy flux (mW/m^2 or Ergs/(cm^2 * s))', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN10-',default_text = '0.2')],
        [sg.CB('TEC graph', key = '-TRUE1-', default = False), sg.CB('SPEC graph', key = '-TRUE2-', default = False), sg.CB('KEOGRAM graph', key = '-TRUE3-', default = False), sg.CB('ROTI graph', key = '-TRUE4-', default = False), sg.CB('Decimation', key = '-TRUE5-', default = False), sg.CB('WT Scaleogram', key = '-TRUE6-', default = False)],
        [sg.CB('SFFT spectrogram', key = '-TRUE7-', default = False), sg.CB('Save Data', key = '-TRUE8-', default = False), sg.CB('Point ROTI', key = '-TRUE9-', default = False),  sg.CB('Mutual Info', key = '-TRUE10-', default = False)],
        [sg.Combo(filters, default_value='butter', key='-FILTER-', enable_events=True)],
        [sg.Output(size = (90,20), key = '-OUTPUT1-')],
        [sg.Submit(), sg.Cancel()]]

window1 = sg.Window('rotida', layout1, grab_anywhere = True, finalize = True)

layout2 = [[sg.Text('Output Folder'), sg.Input(key = '-DIRECTORY2-', enable_events=True), sg.FileBrowse()],
        [sg.Text('Desired Dates (exampe: 01.01.2001 01.02.2001)', size = (40, 1), font = 'Helvetica 10'), sg.InputText(key = '-IN0-')],
        [sg.Output(size=(90,20), key ='-OUTPUT2-')],
        [sg.Submit(), sg.Cancel()]]

window2 = sg.Window('RINEX Downloader', layout2, finalize=True, grab_anywhere=True, location=window1.current_location(), relative_location=(window1.size[0], 0))

while True:
    flaglist = [False for i in range(11)]
    window, event, values = sg.read_all_windows()

    if (window is None) and event != sg.TIMEOUT_EVENT:
        break
    
    if event in (sg.WIN_CLOSED, 'Exit', 'Cancel'):
        window.close()
    if event == '-DIRECTORY1-':
        window['-OUTPUT1-'].update(values['-DIRECTORY1-'])
    if window == window2:
        if event == 'Submit':
            dates = window['-IN0-'].get().split()
            directory = window['-DIRECTORY2-'].get() + '\\'
            ftp_downloader(directory, dates, stations, stations_epmcb, stations_cors_chain)
            
    if event == '-FILTER-':
        filter_name = values['-FILTER-']
         
    if window == window1:
        if event == 'Submit':
            for i in range(1,11):
                if values[f'-TRUE{i}-'] == True:
                    flaglist[i-1] = True
            try:
                bz_path = window['-DIRECTORY3-'].get()
                ae_path = window['-DIRECTORY4-'].get()
                sym_path = window['-DIRECTORY5-'].get()
                output_directory = window['-DIRECTORY6-'].get()
                datez = str(window['-IN0-'].get()).split()
                time_ceil = float(window['-IN2-'].get())
                time_floor = float(window['-IN1-'].get())
                mask_cut_off = float(window['-IN3-'].get())
                satellite_num = float(window['-IN4-'].get())
                window_size = int(window['-IN5-'].get())
                utc_time = str(window['-IN6-'].get())
                lat_window = int(window['-IN7-'].get())
                kp_m = float(window['-IN8-'].get())
                wn = float(window['-IN9-'].get())
                flux = float(window['-IN10-'].get())
            except ValueError:
                print('ValueError')
            main(window['-DIRECTORY1-'].get(), bz_path, ae_path, sym_path, output_directory, datez, window_size, flaglist, satellite_num, wn, utc_time, kp_m, flux, filter_name, mask_cut_off, time_ceil, time_floor, lat_window, stations, stations_epmcb, stations_cors_chain)