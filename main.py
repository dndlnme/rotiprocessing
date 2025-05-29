from datetime import datetime
from utils import graph_roti, graph_tec, plot_graph_tec_path, plot_graph_roti, plot_graph_roti_path, plot_graph_tec, plot_tec_spectrogram, plot_tec_spectrogram_path, plot_tec_spec_once, plot_bz, plot_roti_heatmap, plot_roti_heatmap1, plot_roti_heatmap_eur_am, plot_roti_heatmap_path, plot_wavelet, plot_wavelet_path, plot_sfft_spec, plot_sfft_spec_path, current_roti, current_roti_scatter, plot_current_roti, plot_current_roti_path, MI_plot
from utils import load_bz_data, load_iaga_data, load_data, load_data_path
from utils import process_data, process_data_path, save_data
from utils import butterfilt, cheb, remezfilt, firwinfilt, decimation

filter_functions = {
    'butter': butterfilt,
    'cheby': cheb,
    'remez': remezfilt,
    'firwin': firwinfilt,
    'decimation': decimation
}  


def main(file_path, bz_path, ae_path, sym_path, output_directory, datez, window_size, flaglist, satellite_num, wn, utc_time, kp_m, flux, filter_name, mask_cut_off, time_ceil, time_floor, lat_window, stations, stations_epmcb, stations_cors_chain):
    if len(datez) == 1:
            datez.append(datez[0])
    ds, de = datetime.strptime(datez[0], "%d.%m.%Y"), datetime.strptime(datez[1], "%d.%m.%Y")
    DoYs, DoYe = ds.timetuple().tm_yday, de.timetuple().tm_yday
    bz = load_bz_data(bz_path, DoYs, DoYe) if bz_path != '' else []
    ae = load_iaga_data(ae_path, DoYs, DoYe) if ae_path != '' else []
    sym = load_iaga_data(sym_path, DoYs, DoYe) if sym_path != '' else []
    wn = wn
    
    if file_path.endswith('.Cmn'):
        tec_data = load_data(file_path, mask_cut_off, time_ceil, time_floor, satellite_num, stations)
        processed_data = process_data(tec_data, filter_name, wn, filter_functions)
        
        flag_actions = {
            0: lambda: plot_graph_tec(processed_data, ds),
            1: lambda: plot_tec_spectrogram(processed_data, file_path[file_path.rfind('\\') + 1:-4]),
            2: lambda: plot_roti_heatmap(processed_data, window_size, lat_window, datez),
            3: lambda: plot_graph_roti(processed_data, ds),
            5: lambda: plot_wavelet(processed_data, filter_name, flaglist),
            6: lambda: plot_sfft_spec(processed_data, bz, ae, sym, window_size, DoYs, DoYe, filter_name, flaglist),
            7: lambda: save_data(processed_data, output_directory, file_path, ds, satellite_num, [], filter_name),
            8: lambda: plot_current_roti(utc_time, processed_data, kp_m, ds, flux),
            9: lambda: MI_plot(processed_data, ae, sym, bz)
        }
        
        for index, action in flag_actions.items():
            if flaglist[index]:
                action() 
                
    else:
        tec_data = load_data_path(file_path, mask_cut_off, time_ceil, time_floor, satellite_num, stations)
        processed_data = process_data_path(tec_data, filter_name, wn, filter_functions)
        flag_actions = {
            0: lambda: plot_graph_tec_path(processed_data, ds),
            1: lambda: plot_tec_spectrogram_path(processed_data),
            2: lambda: plot_roti_heatmap_eur_am(processed_data, bz, ae, sym, DoYs, DoYe, window_size, lat_window, ds),
            3: lambda: plot_graph_roti_path(processed_data, ds),
            5: lambda: plot_wavelet_path(processed_data, filter_name, flaglist),
            6: lambda: plot_sfft_spec_path(processed_data, bz, ae, sym, window_size, DoYs, DoYe, filter_name, flaglist),
            8: lambda: plot_current_roti_path(utc_time, processed_data, kp_m, ds, flux)
        }
        # 2: lambda: plot_roti_heatmap_eur_am(processed_data, bz, ae, sym, DoYs, DoYe), 
        # 2: lambda: plot_roti_heatmap_path(processed_data), 
        # Выполнение функций на основе значений флагов
        for index, action in flag_actions.items():
            if flaglist[index]:
                action()        