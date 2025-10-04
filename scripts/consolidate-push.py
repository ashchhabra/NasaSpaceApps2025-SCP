import pandas as pd
from azure.storage.blob import BlobServiceClient
import io
import os

# ---------- Config ----------
CONNECT_STR = "BlobEndpoint=https://datasetsnasa2025.blob.core.windows.net/;QueueEndpoint=https://datasetsnasa2025.queue.core.windows.net/;FileEndpoint=https://datasetsnasa2025.file.core.windows.net/;TableEndpoint=https://datasetsnasa2025.table.core.windows.net/;SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-11-05T01:07:05Z&st=2025-10-04T15:52:05Z&spr=https&sig=pkjW4WcTAalhdlrYIFN2qu14xeLVJPTgYuZWsnnZwVk%3D"
CONTAINER_NAME = "nasa-exoplanets-split"

BLOBS = {
    "k2": {
        "planet": "k2_planets.csv",
        "non_planet": "k2_non_planets.csv"
    },
    "kepler": {
        "planet": "kepler_planets.csv",
        "non_planet": "kepler_non_planets.csv"
    },
    "tess": {
        "planet": "tess_planets.csv",
        "non_planet": "tess_non_planets.csv"
    }
}

blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

def download_blob_to_df(blob_name):
    """Download a CSV blob and load it into a pandas DataFrame"""
    blob_client = container_client.get_blob_client(blob_name)
    stream = blob_client.download_blob().readall()
    return pd.read_csv(io.BytesIO(stream), engine='python', on_bad_lines='skip')

# ---------- Processing Functions ----------
def process_k2():
    df_non = download_blob_to_df(BLOBS['k2']['non_planet'])
    df_non = df_non[['epic_hostname']].rename(columns={'epic_hostname': 'name'})
    df_non['satellite'] = 'K2'
    df_non['label'] = 'non_planet'
    
    df_planet = download_blob_to_df(BLOBS['k2']['planet'])
    if 'epic_hostname' in df_planet.columns:
        df_planet = df_planet[['epic_hostname', 'disposition']].rename(columns={'epic_hostname': 'name'})
    else:
        df_planet = df_planet[['pl_name', 'disposition']].rename(columns={'pl_name': 'name'})
    
    df_planet['label'] = df_planet['disposition'].apply(lambda x: 'planet' if x == 'CONFIRMED' else 'candidate') \
        if 'disposition' in df_planet.columns else 'planet'
    df_planet['satellite'] = 'K2'
    df_planet = df_planet[['name', 'satellite', 'label']]
    
    return pd.concat([df_non, df_planet], ignore_index=True)

def process_kepler():
    df_non = download_blob_to_df(BLOBS['kepler']['non_planet'])
    df_non = df_non[['kepid']].rename(columns={'kepid': 'id'})
    df_non['name'] = 'KIC ' + df_non['id'].astype(str)
    df_non['satellite'] = 'Kepler'
    df_non['label'] = 'non_planet'
    
    df_planet = download_blob_to_df(BLOBS['kepler']['planet'])
    df_planet = df_planet[['kepid', 'disposition']] if 'disposition' in df_planet.columns else df_planet[['kepid']]
    df_planet = df_planet.rename(columns={'kepid': 'id'})
    df_planet['label'] = df_planet['disposition'].apply(lambda x: 'planet' if x == 'CONFIRMED' else 'candidate') \
        if 'disposition' in df_planet.columns else 'planet'
    df_planet['name'] = 'KIC ' + df_planet['id'].astype(str)
    df_planet['satellite'] = 'Kepler'
    df_planet = df_planet[['name', 'satellite', 'label']]
    
    return pd.concat([df_non, df_planet], ignore_index=True)

def process_tess():
    df_non = download_blob_to_df(BLOBS['tess']['non_planet'])
    df_non = df_non[['tid']].rename(columns={'tid': 'id'})
    df_non['name'] = 'TIC ' + df_non['id'].astype(str)
    df_non['satellite'] = 'TESS'
    df_non['label'] = 'non_planet'
    
    df_planet = download_blob_to_df(BLOBS['tess']['planet'])
    df_planet = df_planet[['tid', 'disposition']] if 'disposition' in df_planet.columns else df_planet[['tid']]
    df_planet = df_planet.rename(columns={'tid': 'id'})
    df_planet['label'] = df_planet['disposition'].apply(lambda x: 'planet' if x == 'CONFIRMED' else 'candidate') \
        if 'disposition' in df_planet.columns else 'planet'
    df_planet['name'] = 'TIC ' + df_planet['id'].astype(str)
    df_planet['satellite'] = 'TESS'
    df_planet = df_planet[['name', 'satellite', 'label']]
    
    return pd.concat([df_non, df_planet], ignore_index=True)

# ---------- Combine and Save ----------
df_k2 = process_k2()
df_kepler = process_kepler()
df_tess = process_tess()

df_all = pd.concat([df_k2, df_kepler, df_tess], ignore_index=True)

consolidated_blob = container_client.get_blob_client(
    "consolidated_names_with_planet_candidate_nonplanet.csv"
)
consolidated_blob.upload_blob(df_all.to_csv(index=False).encode(), overwrite=True)

print("Consolidated CSV uploaded to Blob successfully!")