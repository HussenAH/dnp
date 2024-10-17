import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import xarray as xr
import logging
from typing import List
import argparse
from utils.paths import datasets_path
from config import DATA_SOURCE, FROM_YEAR, TO_YEAR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_debug = False
def print_debug(msg):
    if _debug:
        print(msg)

DATA_DIR = os.path.join(datasets_path, DATA_SOURCE, 'raw')

# Ensure DATA_DIR exists
os.makedirs(DATA_DIR, exist_ok=True)


def file_size(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)  # Size in MB


def grib_to_df(grib_file):
    ds = xr.open_dataset(grib_file, engine='cfgrib')
    df = ds.to_dataframe().reset_index()
    df.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'time': 'Time'}, inplace=True)
    return df


def read_pickle_file(filename: str) -> pd.DataFrame:
    try:
        logging.info(f"Reading file: {filename}")
        return pd.read_pickle(filename)
    except Exception as e:
        logging.error(f"Error reading file {filename}: {e}")
        sys.exit(1)


def swap_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_list = list(df)
    col_list = [col_list[0]] + [col_list[2]] + [col_list[1]] + col_list[3:5] + [col_list[6]] + [col_list[5]]
    return df.reindex(columns=col_list).reset_index(drop=True)


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(by=['datetime', 'Longitude', 'Latitude'], axis=0, inplace=True)
    df.set_index(keys=['datetime', 'Longitude', 'Latitude'], drop=True, inplace=True)
    return df


def convert_to_xarray(df: pd.DataFrame) -> xr.Dataset:
    X = df.to_xarray()
    return X.astype(np.float32, casting='same_kind')


def save_to_netcdf(X: xr.Dataset, filedir, filename: str) -> None:
    try:
        netcdf_dir = f"{filedir}/netcdf/"
        os.makedirs(netcdf_dir, exist_ok=True)
        netcdf_filename = f"{netcdf_dir}{filename}"
        logging.info(f"Saving to netCDF: {netcdf_filename}")
        X.to_netcdf(netcdf_filename)
    except Exception as e:
        logging.error(f"Error saving file {filename}: {e}")
        sys.exit(1)


def merge_df(years_list: List[int], output_filename: str, data_directory) -> None:
    if len(years_list) == 0:
        logging.warning(f"No years to merge - {output_filename} not created")
        return
    df_list = [read_pickle_file(f"{data_directory}/{year}_{DATA_SOURCE}_{args.region}.pickle") for year in years_list]
    logging.info("Start concatenating")
    df = pd.concat(df_list)
    logging.info("Finished concatenating")
    logging.info("Swapping columns")
    df = swap_columns(df)
    logging.info("Finished swapping columns")
    logging.info("Start sorting")
    df = sort_dataframe(df)
    logging.info("Finished sorting")
    logging.info("Converting to xarray")
    X = convert_to_xarray(df)
    logging.info("Finished converting to xarray")
    save_to_netcdf(X, data_directory, f"{output_filename}.nc")
    logging.info("Process finished")


def build_value_table_era5(df):
    logging.debug(f"DataFrame columns: {df.columns}")
    df['datetime'] = pd.to_datetime(df['dataDate'].astype(str) + df['dataTime'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
    df.drop(columns=['dataDate', 'dataTime', 'validityDate', 'validityTime'], inplace=True)
    return df

def build_value_table_era5(df):

    print_debug(df.head())
    # Split 'Time' and 'valid_time' into separate date and hour columns
    df[['Date', 'Hour']] = df['Time'].astype(str).str.split(' ', expand=True)
    df[['vDate', 'vHour']] = df['valid_time'].astype(str).str.split(' ', expand=True)

    # Create 'dataDate' and 'dataTime' columns as duplicates of 'Date' and 'Hour'
    df['dataDate'] = df['Date']
    df['dataTime'] = df['Hour']

    # Create 'validityDate' and 'validityTime' columns as duplicates of 'vDate' and 'vHour'
    df['validityDate'] = df['vDate']
    df['validityTime'] = df['vHour']

    # Select and rearrange columns to ensure compatibility with the previous code
    df = df[
        ['dataDate', 'dataTime', 'validityDate', 'validityTime', 'Latitude', 'Longitude', 'u10', 'v10', 't2m', 'sp']]

    # Rename columns to match the required output format
    df.rename(columns={
        't2m': '2t',
        'sp': 'sp',
        'u10': '10u',
        'v10': '10v'
    }, inplace=True)

    return df


def build_data_table_era5(df):
    print_debug("data_table_era5: Building data table\n", df.head())
    df['datetime'] = pd.to_datetime(df['dataDate'].astype(str) + df['dataTime'].astype(str).str.zfill(4), format='%Y%m%d%H%M')
    df.set_index(['datetime', 'Latitude', 'Longitude'], inplace=True)
    df = df.unstack(level=['Latitude', 'Longitude'])
    return df


# def compress_data_table_era5(df, output_pickle):
#     df.to_pickle(output_pickle)
#     print(f"Data table compressed and saved to {output_pickle}")
def compress_data_table_era5(df, output_pickle):
    def compress(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                        start_mem - end_mem) / start_mem))
        return df

    # Read the CSV file
    data = df

    # Control whether the date columns are redundant
    valid_time = data['validityTime'].astype(str).str.strip()
    valid_date = data['validityDate'].astype(str).str.strip()  # astype(str).str.replace('-', '')
    data_time = data['dataTime'].astype(str).str.strip()
    data_date = data['dataDate'].astype(str).str.strip()  # astype(str).str.replace('-', '')

    # Check if the values are equal
    n_inequal = np.sum(np.logical_or(valid_time != data_time, valid_date != data_date))
    if n_inequal > 0:
        sys.exit("Error: n_inequal is not zero.")

    # Remove redundant columns
    data.drop(columns=["dataDate", "dataTime"], inplace=True)
    data.rename(columns={'validityDate': "Date", 'validityTime': "Time"}, inplace=True)

    # Format time column to 4-element string
    data['Time'] = data['Time'].apply(lambda x: '{0:0>4}'.format(x)).astype(str)
    data['Date'] = data['Date'].astype(str)

    # Assuming 'Date' is in 'YYYY-MM-DD' format
    # Convert 'Date' from 'YYYY-MM-DD' to 'YYYYMMDD'
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y%m%d').astype(str)

    # Ensure 'Time' is in HHMM format (assuming it might be in HH:MM:SS)
    # Convert 'Time' to 'HHMM'
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.strftime('%H%M').astype(str)

    # Insert datetime column
    # data.insert(0, "datetime", pd.to_datetime(data['Date'] + data['Time'], format='%Y%m%d%H%M'))
    data.insert(0, "datetime", data['Date'] + data['Time'])

    # Drop time and date
    data.drop(columns=['Time', 'Date'], inplace=True)

    # Order via datetime
    data = data.sort_values("datetime").reset_index(drop=True)
    print_debug(f"data.head()\n {data.head()}")

    # Control whether the number of data points per time is equal
    n_total_grid_points_control = data.groupby("datetime").count()['Latitude'][0]
    any_inequal = (data.groupby("datetime").count() != n_total_grid_points_control).any().any()
    if any_inequal:
        sys.exit("ERROR: The number of points per time is not equal.")

    # Format variables
    data[["2t"]] = data[["2t"]] - 273.15
    data.rename(columns={'2t': "t_in_Cels"}, inplace=True)
    data['sp'] = data['sp'] / 1000
    data.rename(columns={'sp': "sp_in_kPa"}, inplace=True)
    data.rename(columns={'10u': 'wind_10m_north', '10v': 'wind_10m_east'}, inplace=True)

    # Compress the data
    data = compress(data)

    print_debug(f"data.head()\n {data.head()}")
    # Save to pickle
    data.to_pickle(output_pickle)
    print(f"Data table compressed and saved to {output_pickle}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ERA5 data.')
    parser.add_argument('--region', type=str, choices=['US', 'China'], help='Region for the data (US or China)')
    args = parser.parse_args()

    DATA_DIR = os.path.join(DATA_DIR,args.region )

    os.makedirs(DATA_DIR, exist_ok=True)

    data_file_name = f"{DATA_SOURCE}_{args.region}"

    for year in range(FROM_YEAR, TO_YEAR + 1):
        print(f"Year: {year}")
        grib_file = os.path.join(DATA_DIR, f"{year}_{data_file_name}.grib")
        print(f"Size of {grib_file}: {file_size(grib_file):.2f} MB")

        df = grib_to_df(grib_file)
        print("Converted GRIB to DataFrame")

        df = build_value_table_era5(df)
        print("Built value table")

        # TODO: remove this
        #df = build_data_table_era5(df)
        # print("Built data table")

        pickle_file = os.path.join(DATA_DIR, f"{year}_{data_file_name}.pickle")
        compress_data_table_era5(df, pickle_file)
        print(f"Size of {pickle_file}: {file_size(pickle_file):.2f} MB")
        break

    print("Process completed")

    YEARS_TRAIN = []
    YEARS_VALID = []
    YEARS_TEST = []
    for i in range(1986, 2018, 4):
        YEARS_TRAIN = YEARS_TRAIN + [i, i + 1]
        YEARS_VALID.append(i + 2)
        YEARS_TEST.append(i + 3)

    # YEARS_TRAIN = [1980]
    # YEARS_VALID = []
    # YEARS_TEST = []
    merge_df(YEARS_TRAIN, f"train_{DATA_SOURCE}_{args.region}", DATA_DIR)
    merge_df(YEARS_VALID, f"valid_{DATA_SOURCE}_{args.region}", DATA_DIR)
    merge_df(YEARS_TEST, f"test_{DATA_SOURCE}_{args.region}", DATA_DIR)
