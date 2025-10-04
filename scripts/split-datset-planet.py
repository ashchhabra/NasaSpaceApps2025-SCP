import requests
import pandas as pd
from azure.storage.blob import BlobServiceClient
import io

# ---------- Config ----------
CONNECT_STR = "BlobEndpoint=https://datasetsnasa2025.blob.core.windows.net/;QueueEndpoint=https://datasetsnasa2025.queue.core.windows.net/;FileEndpoint=https://datasetsnasa2025.file.core.windows.net/;TableEndpoint=https://datasetsnasa2025.table.core.windows.net/;SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-11-05T01:07:05Z&st=2025-10-04T15:52:05Z&spr=https&sig=pkjW4WcTAalhdlrYIFN2qu14xeLVJPTgYuZWsnnZwVk%3D"
CONTAINER_NAME = "nasa-exoplanets-split"


DATASETS = [
    {
        "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv",
        "satellite": "Kepler",
        "disposition_col": "koi_disposition",
        "planet_values": ["CONFIRMED", "CANDIDATE"],
        "nonplanet_values": ["FALSE POSITIVE"],
        "planet_blob": "kepler_planets.csv",
        "nonplanet_blob": "kepler_non_planets.csv"
    },
    {
        "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv",
        "satellite": "TESS",
        "disposition_col": "tfopwg_disp",
        "planet_values": ["CP","KP","PC"],
        "nonplanet_values": ["FP"],
        "planet_blob": "tess_planets.csv",
        "nonplanet_blob": "tess_non_planets.csv"
    },
    {
        "url": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv",
        "satellite": "K2",
        "disposition_col": "disposition",
        "planet_values": ["CONFIRMED", "CANDIDATE"],
        "nonplanet_values": ["FALSE POSITIVE", "REFUTED"],
        "planet_blob": "k2_planets.csv",
        "nonplanet_blob": "k2_non_planets.csv"
    }

]

blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Create container if it doesn't exist
try:
    container_client.create_container()
except:
    pass

for ds in DATASETS:
    print(f"Processing {ds['satellite']} dataset...")
    
    # Download CSV from NASA URL
    response = requests.get(ds["url"])
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content), engine='python', on_bad_lines='skip')
    
    col = ds["disposition_col"]
    
    # Split into planet / non-planet
    planet_df = df[df[col].isin(ds["planet_values"])]
    nonplanet_df = df[df[col].isin(ds["nonplanet_values"])]
    
    # Upload planet CSV to blob
    planet_blob_client = container_client.get_blob_client(ds["planet_blob"])
    planet_blob_client.upload_blob(
        planet_df.to_csv(index=False).encode(),
        overwrite=True
    )
    
    # Upload non-planet CSV to blob
    nonplanet_blob_client = container_client.get_blob_client(ds["nonplanet_blob"])
    nonplanet_blob_client.upload_blob(
        nonplanet_df.to_csv(index=False).encode(),
        overwrite=True
    )
    
    print(f"{ds['satellite']} dataset uploaded: {len(planet_df)} planets, {len(nonplanet_df)} non-planets.")

print("All datasets processed and uploaded successfully!")