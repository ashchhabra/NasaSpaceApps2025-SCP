import lightkurve as lk
import pandas as pd
import os

# Load CSV with 'name' column
csv_path = '../consolidated_names_with_planet_candidate_nonplanet.csv'  # adjust path
df = pd.read_csv(csv_path)

# Directory to save light curves
save_dir = 'downloaded_lightcurves'
os.makedirs(save_dir, exist_ok=True)

# List to hold paths or summaries of downloaded files per target
download_paths = []

for idx, row in df.iterrows():
    target_name = row['name']
    satellite = row['satellite']
    print(f"Downloading light curve for {target_name} ({satellite})...")

    try:
        search_result = lk.search_lightcurve(target_name)

       # Filter based on satellite to exclude unsupported/lightcurve types
        if satellite == 'K2':
            # Exclude K2VARCAT and any other unsupported types here
            search_result = search_result[~search_result['dataproduct_type'].isin(['K2VARCAT'])]
        # elif satellite == 'Kepler':
            # Optionally filter Kepler products if needed
            # search_result = search_result[~search_result['dataproduct_type'].isin(['LIGHTCURVE_FFI'])]  # example filter
        # elif satellite == 'TESS':
            # Filter out low quality or undesired TESS products if desired
            # search_result = search_result[~search_result['dataproduct_type'].isin(['FFI'])]  # example filter
        if len(search_result) == 0:
            print(f"No supported light curve found for {target_name}")
            download_paths.append('')
            continue
        
        lc_collection = search_result.download()

        saved_files = []
        for i, lc in enumerate(lc_collection):
            clean_name = target_name.replace(" ", "_")
            filename = f"{clean_name}_{i}.fits"
            filepath = os.path.join(save_dir, filename)
            lc.to_fits(filepath)
            saved_files.append(filepath)

        # Join multiple file paths with semicolon
        download_paths.append(';'.join(saved_files))

    except Exception as e:
        print(f"Error downloading {target_name}: {e}")
        download_paths.append('')

# Add new column with paths to downloaded files
df['downloaded_lightcurves'] = download_paths

# Save updated CSV
df.to_csv('csv_with_references.csv', index=False)
