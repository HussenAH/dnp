import cdsapi
import os, sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)
from utils.paths import datasets_path

out_path = os.path.join(datasets_path, 'ERA5', 'raw', 'Chaina')
# Ensure the data directory exists
os.makedirs(out_path, exist_ok=True)
raise Exception("This script is not working. It is not clear what the error is")
c = cdsapi.Client()

for year in range(1980,2019):
    #Get year as string and set filename:
    yearname=str(year)
    filename="data/"+yearname+"_ERA5_China.grib"
    #Download the data of that year:
    c.retrieve(
        'reanalysis-era5_dataset-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
                '10m_v_component_of_wind', '2m_temperature', 'orography',
                'surface_pressure',
            ],
            'year': yearname,
            'month': [
                '01', '02', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                35, 105, 25,
                115,
            ],
        },
        filename)
