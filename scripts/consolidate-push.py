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
# def process_k2():
#     df_non = download_blob_to_df(BLOBS['k2']['non_planet'])
#     df_non = df_non[['epic_hostname']].rename(columns={'epic_hostname': 'name'})
#     df_non['satellite'] = 'K2'
#     df_non['label'] = 'non_planet'
    
#     df_planet = download_blob_to_df(BLOBS['k2']['planet'])
#     if 'epic_hostname' in df_planet.columns:
#         df_planet = df_planet[['epic_hostname', 'disposition']].rename(columns={'epic_hostname': 'name'})
#     else:
#         df_planet = df_planet[['pl_name', 'disposition']].rename(columns={'pl_name': 'name'})
    
#     df_planet['label'] = df_planet['disposition'].apply(lambda x: 'planet' if x == 'CONFIRMED' else 'candidate') \
#         if 'disposition' in df_planet.columns else 'planet'
#     df_planet['satellite'] = 'K2'
#     df_planet = df_planet[['name', 'satellite', 'label']]
    
#     return pd.concat([df_non, df_planet], ignore_index=True)

def process_kepler():
    # Load planet data
    df_planet = download_blob_to_df(BLOBS['kepler']['planet'])

    # Define the column mapping from Kepler catalog columns to your feature names
    column_map = {
        'koi_prad': 'planet_radii',
        'koi_depth': 'transit_depth',
        'koi_period': 'days',
        'koi_srad': 'star_radii',
        'koi_insol': 'earth_flux',
        'koi_steff': 'star_temp'
    }

    # Keep only the relevant columns + disposition + kepid
    cols_to_keep = ['kepid', 'koi_disposition'] + list(column_map.keys())
    available_cols = [col for col in cols_to_keep if col in df_planet.columns]

    df_planet_sub = df_planet[available_cols].copy()

    # Rename columns to your feature names
    rename_dict = {'kepid': 'id'}
    rename_dict.update({k: v for k, v in column_map.items() if k in df_planet_sub.columns})
    df_planet_sub = df_planet_sub.rename(columns=rename_dict)

    # Map disposition to numeric labels
    label_map = {
        'CONFIRMED': 2,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    }

    if 'koi_disposition' in df_planet_sub.columns:
        df_planet_sub['label'] = df_planet_sub['koi_disposition'].str.upper().map(label_map).fillna(1).astype(int)
        df_planet_sub = df_planet_sub.drop(columns=['koi_disposition'])
    else:
        df_planet_sub['label'] = 2  # default label

    # Create "name" column from id for consistency
    df_planet_sub['name'] = 'KIC ' + df_planet_sub['id'].astype(str)

    # Drop rows with any NaN or null values in these columns
    cols_to_check = [
        'planet_radii', 'transit_depth', 'days',
        'star_radii', 'earth_flux', 'star_temp', 'label'
    ]
    df_planet_sub = df_planet_sub.dropna(subset=[col for col in cols_to_check if col in df_planet_sub.columns])

    # Final columns order
    final_cols = [
        'name', 'label',
        'planet_radii', 'transit_depth', 'days',
        'star_radii', 'earth_flux', 'star_temp'
    ]

    # Some columns might be missing; keep only those present
    final_cols_available = [col for col in final_cols if col in df_planet_sub.columns]

    df_planet_sub = df_planet_sub[final_cols_available]

    return df_planet_sub


def process_tess():

    # Load planet data
    df_planet = download_blob_to_df(BLOBS['tess']['planet'])

    # Define the column mapping from TESS columns to your feature names
    column_map = {
        'pl_orbper': 'days',
        'pl_trandep': 'transit_depth',
        'pl_rade': 'planet_radii',
        'pl_insol': 'earth_flux',
        'st_rad': 'star_radii',
        'st_teff': 'star_temp'  # Added stellar effective temperature
    }

    # Select relevant columns
    cols_to_keep = ['tid', 'tfopwg_disp'] + list(column_map.keys())
    available_cols = [col for col in cols_to_keep if col in df_planet.columns]

    df_planet_sub = df_planet[available_cols].copy()

    # Rename columns to standardized names
    rename_dict = {'tid': 'id'}
    rename_dict.update({k: v for k, v in column_map.items() if k in df_planet_sub.columns})
    df_planet_sub = df_planet_sub.rename(columns=rename_dict)

    # Map disposition to numeric label
    label_map = {
        'CP': 2,  # Confirmed Planet
        'PC': 1,  # Potential Candidate
        'FP': 0,  # False Positive
        'KP': 2   # Known Planet, treated as confirmed
    }

    if 'tfopwg_disp' in df_planet_sub.columns:
        df_planet_sub['label'] = df_planet_sub['tfopwg_disp'].str.upper().map(label_map).fillna(1).astype(int)
        df_planet_sub = df_planet_sub.drop(columns=['tfopwg_disp'])
    else:
        df_planet_sub['label'] = 2  # Default label if disposition missing

    # Add name and satellite columns
    df_planet_sub['name'] = 'TIC ' + df_planet_sub['id'].astype(str)

    # Drop rows with any NaN or null values in these columns
    cols_to_check = [
        'planet_radii', 'transit_depth', 'days',
        'star_radii', 'earth_flux', 'star_temp', 'label'
    ]
    df_planet_sub = df_planet_sub.dropna(subset=[col for col in cols_to_check if col in df_planet_sub.columns])

    # Final columns for output
    final_cols = [
        'name', 'label',
        'planet_radii', 'transit_depth', 'days',
        'star_radii', 'earth_flux', 'star_temp'
    ]
    final_cols_available = [col for col in final_cols if col in df_planet_sub.columns]

    df_planet_sub = df_planet_sub[final_cols_available]

    return df_planet_sub



# ---------- Combine and Save ----------
# df_k2 = process_k2()
df_kepler = process_kepler()
df_tess = process_tess()

df_all = pd.concat([df_kepler, df_tess], ignore_index=True)

consolidated_blob = container_client.get_blob_client(
    "consolidated_names_with_planet_candidate_nonplanet.csv"
)
consolidated_blob.upload_blob(df_all.to_csv(index=False).encode(), overwrite=True)

print("Consolidated CSV uploaded to Blob successfully!")