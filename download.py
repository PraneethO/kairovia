import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from meteostat import Stations, Hourly
import os

cache_dir = os.path.expanduser("~/.meteostat/cache")
if os.path.exists(cache_dir):
    import shutil
    try:
        shutil.rmtree(cache_dir)
        print("Cleared meteostat cache")
    except:
        print("Could not clear cache")

today_utc = datetime.now(timezone.utc)
start_utc = today_utc - relativedelta(years=20) # change this for time to likke days=1 or years=5

today = today_utc.replace(tzinfo=None)
start = start_utc.replace(tzinfo=None)

print(f"Fetching Austin weather data from {start} to {today}")

try:
    STATION_ID = '74745'  # meteostat id for austin
    station_name = "Austin / Del Valle (KAUS)" # not actually neccesary
    
    print(f"Using KAUS station: {STATION_ID} ({station_name})")
    
    print(f"Using station: {STATION_ID}")

    data = Hourly(STATION_ID, start, today)
    df = data.fetch()
    if df.empty:
        raise RuntimeError(f"No hourly data returned for station {STATION_ID}.")

    print(f"Fetched {len(df)} rows from {STATION_ID}")
    
except Exception as e:
    print(f"Error fetching weather data: {e}")
    raise
print(df)

df['pres'] = df['pres'].ffill().bfill()

if df['pres'].isna().any():
    df['pres'] = df['pres'].fillna(1013.25)
    print("Warning: Some pressure data was missing and filled with standard atmospheric pressure")

df = df[['temp', 'dwpt', 'pres', 'wdir', 'wspd', 'prcp']]
df.rename(columns={
    'temp': 'TMP',
    'dwpt': 'DEWP',
    'pres': 'SLP',
    'wdir': 'WDIR',
    'wspd': 'WSPD',
    'prcp': 'PRCP'
}, inplace=True)

df.index.name = 'DATE'
output_file = "data.csv"
df.to_csv(output_file)
print(f"Saved {len(df)} rows to {output_file}")
