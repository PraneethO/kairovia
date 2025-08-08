import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from meteostat import Stations, Hourly

# helper script to look for station IDs around austin or whatever coordinates
austin_stations = Stations()
austin_stations = austin_stations.nearby(30.1975, -97.6664)  # Austin Bergstrom airport coordinates
austin_data = austin_stations.fetch(10)

print("Austin Bergstrom area stations:")
print(austin_data[['name', 'wmo', 'icao', 'hourly_start', 'hourly_end']].head(10))

print("\n Testing austin stations")
today_utc = datetime.now(timezone.utc)
start_utc = today_utc - relativedelta(days=2)
today = today_utc.replace(tzinfo=None)
start = start_utc.replace(tzinfo=None)

for station_id in austin_data.index[:5]:
    try:
        print(f"\nTesting {station_id}: {austin_data.loc[station_id, 'name']}")
        data = Hourly(station_id, start, today)
        df = data.fetch()
        if not df.empty:
            print(f"  ✓ SUCCESS: {len(df)} rows of data available")
            print(f"  Latest data: {df.index[-1]}")
            print(f"  ICAO: {austin_data.loc[station_id, 'icao']}")
            print(f"  WMO: {austin_data.loc[station_id, 'wmo']}")
            break
        else:
            print(f"  ✗ No data")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*60)
