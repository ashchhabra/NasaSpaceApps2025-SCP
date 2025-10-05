import sys
import os
import numpy as np
import pandas as pd
import lightkurve as lk
from tsfresh import extract_relevant_features
import inspect
from tsfresh.feature_extraction import feature_calculators


# ==================== CONFIGURE BATCH/PATHS ===================
BATCH_SIZE = 100  # Smaller batch size for faster processing and smaller files
CACHE_DIR = "lightcurve_batches"
FEATURES_DIR = "feature_batches"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)


# ================== HELPERS FOR MISSIONS ======================
def detect_mission_author(star_name):
    if star_name.startswith('TIC'):
        return "TESS", "SPOC"
    elif star_name.startswith('KIC'):
        return "Kepler", "Kepler"
    elif star_name.startswith('EPIC'):
        return "K2", "K2"
    else:
        return None, None


# ============= UNIFORM SAMPLING AND IMPUTATION ================
def uniform_resample_and_impute(time, flux):
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[valid_mask]
    flux = flux[valid_mask]
    time_regular = np.arange(time.min(), time.max(), 1/24).astype(np.float32)
    flux_regular = np.interp(time_regular, time, flux).astype(np.float32)
    df = pd.DataFrame({'time': time_regular, 'flux': flux_regular})
    df['flux'] = df['flux'].interpolate(method='linear', limit_direction='both')
    df['flux'] = df['flux'].bfill().ffill()
    return df


# ============= FETCH AND PREPROCESS LIGHT CURVES ==============
def get_lightcurve_df(star_name):
    mission, author = detect_mission_author(star_name)
    if not mission:
        print(f"Unknown mission for input: {star_name}")
        return None
    try:
        search_result = lk.search_lightcurve(star_name, mission=mission)
        if len(search_result) > 0:
            if "product" in search_result.table.colnames:
                mask = (search_result.author == author) & (
                    search_result.table["product"] == "PDCSAP_FLUX")
            else:
                mask = (search_result.author == author)
            filtered = search_result[mask]
            if len(filtered) > 0:
                lc = filtered[0].download(quality_bitmask='hard')
            else:
                print(f"No {author} PDCSAP_FLUX products found. Using first available light curve.")
                lc = search_result[0].download(quality_bitmask='default')
            flattened_lc = lc.flatten()
            time = np.ascontiguousarray(np.asarray(flattened_lc.time.value, dtype=np.float32))
            flux = np.ascontiguousarray(np.asarray(flattened_lc.flux.value, dtype=np.float32))
            flux = (flux - np.nanmedian(flux)) / np.nanstd(flux)
            mask_clip = np.abs(flux) < 5
            time = time[mask_clip]
            flux = flux[mask_clip]
            df = uniform_resample_and_impute(time, flux)
            return df
        else:
            print(f"No light curves found for {star_name}.")
            return None
    except Exception as e:
        print(f"An error occurred for {star_name}: {e}")
        return None


# =========== FEATURE DOCUMENTATION AUTO-EXPORT ===========
def get_tsfresh_feature_docs(feature_columns):
    feat_docs = {}
    for name, func in feature_calculators.__dict__.items():
        if callable(func) and not name.startswith('_'):
            doc = inspect.getdoc(func)
            if doc is not None:
                doc = doc.replace('\n', ' ').strip()
            feat_docs[name] = doc
    rows = []
    for col in feature_columns:
        base_name = col.split('__')[0]
        rows.append({
            'feature': col,
            'description': feat_docs.get(base_name, '')
        })
    return pd.DataFrame(rows)


def save_feature_doc_to_readme(features_df, filename="README.md"):
    doc_df = get_tsfresh_feature_docs(features_df.columns)
    with open(filename, 'w') as f:
        f.write("# TSFresh Feature Documentation\n\n")
        f.write("| Feature Name | Description |\n")
        f.write("|--------------|-------------|\n")
        for _, row in doc_df.iterrows():
            desc = row['description'] if row['description'] else "No official docstring found."
            f.write(f"| `{row['feature']}` | {desc} |\n")
    print(f"Feature documentation saved to {filename}")


# ============ BATCH-PROCESSING LOOP =====================
def process_and_save_batch(df_planets, batch_number, start_idx, end_idx):
    all_lc_rows = []
    y = {}
    failed_ids = set()
    batch_total = end_idx - start_idx
    print(f"\nProcessing batch {batch_number}: rows {start_idx} to {end_idx-1} (total: {batch_total})")

    for j, i in enumerate(range(start_idx, end_idx)):
        row = df_planets.iloc[i]
        star_name = row['name']
        label = row['label']
        print(f"[Batch {batch_number} | {j+1}/{batch_total} | CSV idx {i}] Star: {star_name}")
        lc_df = get_lightcurve_df(star_name)
        if lc_df is not None and not lc_df.isnull().values.any():
            lc_df = lc_df.dropna(subset=["flux", "time"])
            lc_df['id'] = i
            all_lc_rows.append(lc_df)
            y[i] = label
        else:
            print(f"-> Skipping {star_name} (CSV idx {i}): no valid light curve")
            failed_ids.add(i)

    if not all_lc_rows:
        print(f"No valid light curves processed in batch {batch_number}.")
        return None, None

    all_lc_df = pd.concat(all_lc_rows, ignore_index=True)
    all_lc_df.to_parquet(os.path.join(CACHE_DIR, f"lc_batch_{batch_number}.parquet"))

    y_series = pd.Series(y)

    print(f"\nExtracting TSFresh features for batch {batch_number} ({batch_total} processed)...")
    features = extract_relevant_features(
        all_lc_df,
        y_series,
        column_id='id',
        column_sort='time',
        n_jobs=8
    )
    
    # Align features with the batch DataFrame and drop failed rows from the original batch
    batch_orig = df_planets.iloc[start_idx:end_idx].drop(index=list(failed_ids), errors='ignore')
    # Ensure features and batch_orig align on index
    features = features.loc[batch_orig.index.intersection(features.index)]
    
    # Join original data columns with extracted features
    batch_with_features = batch_orig.join(features, how='left')

    # Save combined batch CSV file with original columns and features
    batch_csv_path = os.path.join(FEATURES_DIR, f"features_batch_{batch_number}.csv")
    batch_with_features.to_csv(batch_csv_path, index=False)
    print(f"Saved batch {batch_number} CSV with features to {batch_csv_path}")

    return features, failed_ids


# ================== MAIN DRIVER =========================
def main():
    input_csv = './consolidated.csv'
    df_planets = pd.read_csv(input_csv)
    n = len(df_planets)
    all_failed_ids = set()

    for batch_start in range(0, n, BATCH_SIZE):
        batch_number = batch_start // BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, n)
        print(f"\n====== Starting batch {batch_number} ({batch_start} to {batch_end-1}) ======")
        _, failed_ids = process_and_save_batch(df_planets, batch_number, batch_start, batch_end)
        all_failed_ids.update(failed_ids)

    print(f"\nProcessing complete. Total failed IDs: {len(all_failed_ids)}")

    # Optional: create master CSV by concatenating batch CSVs (if desired)
    # You can implement this if you want a combined CSV after batch processing

if __name__ == "__main__":
    main()
