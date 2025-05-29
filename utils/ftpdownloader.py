# скачивание rinex файлов (пока что без обработки)
from ftplib import FTP, FTP_TLS
import os
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
import requests
    
# stations = ['svtl', 'zeck', 'krs1', 'glsv', 'mdvj', 'auto', 'SEPT', 'POLV', 'KIRU', 'LAMA', 'RIGA', 'UZHL','BUCU', 'kiru', 'bucu', 'polv', 'riga', 'uzhl'] # список станций (лучше вынести в txt)

# def download_from_ftp_server(dates):
#     if len(dates) == 1:
#         dates.append(dates[0])
#     date_start, date_end = datetime.strptime(dates[0], "%d.%m.%Y"), datetime.strptime(dates[1], "%d.%m.%Y")
#     day_start, day_end = date_start.timetuple().tm_yday, date_end.timetuple().tm_yday
#     year, month_start, month_end = str(date_start.year), date_start.month, date_end.month
#     months = [str(i).zfill(2) for i in range(month_start, month_end + 1)]

#     ftp_host = 'garner.ucsd.edu' # данные по станциям
#     remote_directory = f'/pub/rinex/{year}/'
#     local_directory = os.path.join(directory, f'data{year[-2:]}')
#     os.makedirs(local_directory, exist_ok=True)

#     ftp = FTP(ftp_host)
#     ftp.login()  # при необходмости ftp.login('username', 'password') 
#     ftp.cwd(remote_directory)
#     days = ftp.nlst()

#     for day in days[day_start-1:day_end]:
#         try:
#             ftp.cwd(day) 
#             raw_data = ftp.nlst()   
#             for station in stations:
#                 for data in raw_data:
#                     if data.startswith(station) and (data.endswith(f'{year[-2:]}d.Z') or 
#                     data.endswith(f'{year[-2:]}n.Z') or data.endswith(f'crx.gz')):
#                         local_filename = os.path.join(local_directory, data)
#                         print(f'Downloading {data} to {local_filename}')
#                         with open(local_filename, 'wb') as file:
#                             ftp.retrbinary(f'RETR {data}', file.write)
            
#             ftp.cwd('..')
        
#         except Exception as e:
#             print(f"An error occurred while processing {day}: {e}")

#     ftp.quit()

def download_from_ftp_server(directory, dates, stations, stations_epmcb, stations_cors_chain):
    if len(dates) == 1:
        dates.append(dates[0])
    date_start, date_end = datetime.strptime(dates[0], "%d.%m.%Y"), datetime.strptime(dates[1], "%d.%m.%Y")
    day_start, day_end = date_start.timetuple().tm_yday, date_end.timetuple().tm_yday
    year, month_start, month_end = str(date_start.year), date_start.month, date_end.month
    months = [str(i).zfill(2) for i in range(month_start, month_end + 1)]

    ftp_host = 'gdc.cddis.eosdis.nasa.gov' # данные по станциям
    remote_directory = f'/gnss/data/daily/{year}/'
    # remote_directory = r'/archive/gnss/data/daily/{year}/'
    local_directory = os.path.join(directory, f'data{year[-2:]}')
    os.makedirs(local_directory, exist_ok=True)
    ftp = FTP_TLS(host = ftp_host)
    # ftp.debugging = 2
    ftp.login('anonymous', 'danymor03@gmail.com')  # при необходмости ftp.login('username', 'password') 
    ftp.prot_p()
    ftp.cwd(remote_directory)
    days = ftp.nlst()
    for day in days[day_start-1:day_end]:
        try:
            ftp.cwd(day) 
            ftp.cwd(f'{year[-2:]}d')
            raw_data = ftp.nlst()
            for station in stations:
                for data in raw_data:
                    if (data.startswith(station) or data.startswith(station.lower())) and (data.endswith(f'{year[-2:]}d.gz') or data.endswith(f'crx.gz') or data.endswith(f'{year[-2:]}d.Z')):
                        local_filename = os.path.join(local_directory, data)
                        print(f'Downloading {data} to {local_filename}')
                        with open(local_filename, 'wb') as file:
                            ftp.retrbinary(f'RETR {data}', file.write)
            ftp.cwd(f'/gnss/data/daily/{year}/')
        except Exception as e:
            print(f"An error occurred while processing {day}: {e}")
        
    for day in days[day_start-1:day_end]:
        try:
            ftp.cwd(day) 
            ftp.cwd(f'{year[-2:]}n')
            raw_data = ftp.nlst()   
            for station in stations:
                for data in raw_data:
                    if (data.startswith(station) and (data.endswith(f'{year[-2:]}n.gz') or data.endswith(f'{year[-2:]}n.Z') or data.endswith(f'{year[-2:]}n.z') or data.endswith(f'rnx.Z'))):
                        local_filename = os.path.join(local_directory, data)
                        print(f'Downloading {data} to {local_filename}')
                        with open(local_filename, 'wb') as file:
                            ftp.retrbinary(f'RETR {data}', file.write)
        except Exception as e:
            print(f"An error occurred while processing {day}: {e}")
    ftp.quit()
    
    ftp_host = 'epncb.oma.be'
    remote_directory = f'/pub/RINEX/{year}'
    local_directory = os.path.join(directory, f'data{year[-2:]}')
    os.makedirs(local_directory, exist_ok=True) 
    ftp = FTP(host=ftp_host)
    ftp.login()
    ftp.cwd(remote_directory)
    days = ftp.nlst()
    for day in days[day_start-1:day_end]:
        try:
            ftp.cwd(day) 
            raw_data = ftp.nlst()
            for station in stations_epmcb:
                for data in raw_data:
                    if (data.startswith(station) or data.startswith(station.lower())):
                        local_filename = os.path.join(local_directory, data)
                        print(f'Downloading {data} to {local_filename}')
                        with open(local_filename, 'wb') as file:
                            ftp.retrbinary(f'RETR {data}', file.write)         
            ftp.cwd(f'/pub/RINEX/{year}/')
        except Exception as e:
            print(f"An error occurred while processing {day}: {e}")
    ftp.quit()
    
    ftp_host = 'ftp.chain-project.net'
    remote_directory = f'/gps/data/daily/{year}/'
    local_directory = os.path.join(directory, f'data{year[-2:]}')
    os.makedirs(local_directory, exist_ok=True)
    ftp = FTP(host=ftp_host)
    ftp.login('ftp', 'st094271@student.spbu.ru')
    ftp.cwd(remote_directory)
    days = ftp.nlst()
    for day in days[day_start-1:day_end]:
        try:
            ftp.cwd(f'{day}/{year[-2:]}d')
            raw_data = ftp.nlst()
            print(raw_data)
            for station in stations_cors_chain:
                for data in raw_data:
                    if (data.startswith(station) or data.startswith(station.lower())):
                        local_filename = os.path.join(local_directory, data)
                        print(f'Downloading {data} to {local_filename}')
                        with open(local_filename, 'wb') as file:
                            ftp.retrbinary(f'RETR {data}', file.write)  
            ftp.cwd(f'/gps/data/daily/{year}/')  
        except Exception as e:
            print(f"An error occurred while processing {day}: {e}")
    ftp.quit()
    
    https_host = 'https://geodesy.noaa.gov'
    remote_directory = f'/corsdata/rinex/{year}/'
    local_directory = os.path.join(directory, f'data{year[-2:]}')
    os.makedirs(local_directory, exist_ok=True)
    links = [
    f"{https_host}{remote_directory}{str(day).zfill(3)}/{station.lower()}/{station.lower()}{str(day).zfill(3)}0.{year[-2:]}d.gz"
    for day in range(day_start, day_end + 1)
    for station in stations
    ]
    for url in links:
        try:
            filename = url.split('/')[-1]
            local_path = os.path.join(local_directory, filename)    
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Успешно скачан: {filename}")
            else:
                print(f"Файл не найден: {url}")
        except Exception as e:
            print(f"Ошибка при скачивании {url}: {str(e)}")
    
    ftp_host = 'ftp.aiub.unibe.ch' # данные по P1C1 P1P2
    remote_directory = f'/CODE/{year}/'
    os.makedirs(local_directory, exist_ok=True)

    ftp = FTP(ftp_host)
    ftp.login()  # при необходмости ftp.login('username', 'password') 
    ftp.cwd(remote_directory)
    PCs = ftp.nlst()

    for PC in PCs:
        for month in months:
            if (PC == f'P1C1{year[-2:]}{month}.DCB.Z') or (PC == f'P1P2{year[-2:]}{month}.DCB.Z'):
                local_filename = os.path.join(local_directory, PC)
                print(f'Downloading {PC} to {local_filename}')
                with open(local_filename, 'wb') as file:
                    ftp.retrbinary(f'RETR {PC}', file.write)
    ftp.quit()
    
    ftp_host = 'spdf.gsfc.nasa.gov'
    remote_directory = '/pub/data/omni/high_res_omni/monthly_1min/'
    
    os.makedirs(local_directory, exist_ok=True)

    try:
        ftp = FTP_TLS(ftp_host)
        ftp.login() 
        ftp.cwd(remote_directory)
        omni_month = ftp.nlst()

        for omni in omni_month:
            try:
                if f'{year}' in omni:
                    for month in months:
                        if f'{year}{month}' in omni:
                            local_filename = os.path.join(local_directory, os.path.basename(omni))
                            print(f'Downloading {omni} to {local_filename}')
                            with open(local_filename, 'wb') as file:
                                ftp.retrbinary(f'RETR {omni}', file.write)
                            break 

            except Exception as e:
                print(f"An error occurred while processing {omni}: {e}")

    except Exception as e:
        print(f"Failed to connect or process FTP: {e}")
    
    finally:
        ftp.quit()
        
    # ftp_host = 'ftp.ngdc.noaa.gov'
    # dir = '/STP/GEOMAGNETIC_DATA/INDICES/AURORAL_ELECTROJET/ONE_MINUTE'
    # ftp = FTP(ftp_host)
    # ftp.login()
    # ftp.cwd(dir)
    # print(ftp.nlst())
    # тут тоже АЕ до 2011 есть
        
    return local_directory

def unzip(local_directory):
    output_directory = local_directory

    for filename in os.listdir(local_directory):
        p = Path(local_directory)
        if (filename.endswith('z')) or filename.endswith('Z'):
            
            file_path = p / filename
            print(f'Unpacking: {file_path}')
            
            try:
                result = subprocess.run(['c:/Program Files/7-Zip/7z.exe', 'e', file_path, f'-o{output_directory}'], check=True, capture_output=True, text=True, timeout = 30)
                # c:/Program Files/7-Zip/7z.exe /Users/daniilmorgunov/Downloads/7z2408-mac (1) 2/7zz
                print(f'Successfully unpacked {file_path}')
                os.remove(file_path)
                print(f'Deleted archive file: {file_path}')
                
            except FileNotFoundError as e:
                print(f'Error: {e}. Check if "7z" is installed and in your PATH.')
            except subprocess.CalledProcessError as e:
                print(f'Failed to unpack {file_path}. Error: {e}')

def replace_cmns(output_directory):
    CMNs = os.listdir(output_directory)
    processed_directory = output_directory + '/processed_data'
    for CMN in CMNs:
        if CMN.endswith('.Cmn'):
            output_file = os.path.join(output_directory, CMN)
            input_file = os.path.join(processed_directory, CMN)
            os.replace(output_file, input_file)
            
def send_to_gps_tec(output_directory, directory):
    processed_directory = output_directory + '/processed_data'
    exe_path = r"D:\data\GPS_Gopi_v3.0\GPS_Gopi_v3.5\GPS_TEC.exe"
    os.makedirs(processed_directory, exist_ok = True)
    rinexes = os.listdir(output_directory)
    for rinex in rinexes:
        p = Path(directory)
        if rinex.endswith('d'):
            # rinex_path = p / rinex
            rinex_path = os.path.join(directory, rinex)
            try:
                result = subprocess.run([exe_path, rinex_path, 'auto'], check = False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("Output:", result.stdout.decode())
                output_file = os.path.join(processed_directory, rinex)
                with open(output_file, 'w') as f:
                    f.write(rinex)
            except subprocess.CalledProcessError as e:
                print("An error occurred while running the executable:", e)
                print("Error Output:", e.stderr.decode())
            break

def processed_output(output_directory):
    CMNs = os.listdir(output_directory)
    processed_directory = output_directory + '/processed_data'
    for CMN in CMNs:
        if CMN.endswith('.Cmn'):
            output_file = os.path.join(output_directory, CMN)
            input_file = os.path.join(processed_directory, CMN)
            os.replace(output_file, input_file)
            
def ftp_downloader(directory, dates, stations, stations_epmcb, stations_cors_chain):
        local_directory = download_from_ftp_server(directory, dates, stations, stations_epmcb, stations_cors_chain)
        unzip(local_directory)
        replace_cmns(local_directory)
        # send_to_gps_tec(directory, local_directory)
        # processed_output(directory)