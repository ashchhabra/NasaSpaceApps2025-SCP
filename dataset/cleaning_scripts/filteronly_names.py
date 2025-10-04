import os
import pandas as pd


def process_k2():
    df_non = pd.read_csv('data/k2/non_planet/k2_non_planet.csv')
    df_non = df_non[['epic_hostname']].rename(columns={'epic_hostname': 'name'})
    df_non['satellite'] = 'K2'
    df_non['label'] = 'non_planet'
    
    df_planet = pd.read_csv('data/k2/planet/k2_planets.csv')
    # Use epic_hostname if available, else fall back to pl_name
    if 'epic_hostname' in df_planet.columns:
        df_planet = df_planet[['epic_hostname', 'disposition']].rename(columns={'epic_hostname': 'name'})
    else:
        df_planet = df_planet[['pl_name', 'disposition']].rename(columns={'pl_name': 'name'})

    # Assign labels based on disposition
    if 'disposition' in df_planet.columns:
        df_planet['label'] = df_planet['disposition'].apply(lambda x: 'planet' if x == 'CONFIRMED' else 'candidate')
    else:
        df_planet['label'] = 'planet'

    df_planet['satellite'] = 'K2'
    df_planet = df_planet[['name', 'satellite', 'label']]

    return pd.concat([df_non, df_planet], ignore_index=True)


def process_kepler():
    df_non = pd.read_csv('data/kepler/non_planet/kepler_non_planet.csv')
    df_non = df_non[['kepid']].rename(columns={'kepid': 'id'})
    df_non['name'] = 'KIC ' + df_non['id'].astype(str)
    df_non['satellite'] = 'Kepler'
    df_non['label'] = 'non_planet'
    
    df_planet = pd.read_csv('data/kepler/planet/kepler_planets.csv')
    df_planet = df_planet[['kepid', 'disposition']] if 'disposition' in df_planet.columns else df_planet[['kepid']]
    df_planet = df_planet.rename(columns={'kepid': 'id'})

    if 'disposition' in df_planet.columns:
        df_planet['label'] = df_planet['disposition'].apply(lambda x: 'planet' if x == 'CONFIRMED' else 'candidate')
    else:
        df_planet['label'] = 'planet'

    df_planet['name'] = 'KIC ' + df_planet['id'].astype(str)
    df_planet['satellite'] = 'Kepler'
    df_planet = df_planet[['name', 'satellite', 'label']]

    return pd.concat([df_non, df_planet], ignore_index=True)


def process_tess():
    df_non = pd.read_csv('data/tess/non_planet/tess_non_planet.csv')
    df_non = df_non[['tid']].rename(columns={'tid': 'id'})
    df_non['name'] = 'TIC ' + df_non['id'].astype(str)
    df_non['satellite'] = 'TESS'
    df_non['label'] = 'non_planet'
    
    df_planet = pd.read_csv('data/tess/planet/tess_planets.csv')
    df_planet = df_planet[['tid', 'disposition']] if 'disposition' in df_planet.columns else df_planet[['tid']]
    df_planet = df_planet.rename(columns={'tid': 'id'})

    if 'disposition' in df_planet.columns:
        df_planet['label'] = df_planet['disposition'].apply(lambda x: 'planet' if x == 'CONFIRMED' else 'candidate')
    else:
        df_planet['label'] = 'planet'

    df_planet['name'] = 'TIC ' + df_planet['id'].astype(str)
    df_planet['satellite'] = 'TESS'
    df_planet = df_planet[['name', 'satellite', 'label']]

    return pd.concat([df_non, df_planet], ignore_index=True)


# Combine all
df_k2 = process_k2()
df_kepler = process_kepler()
df_tess = process_tess()

df_all = pd.concat([df_k2, df_kepler, df_tess], ignore_index=True)

os.makedirs('data/justnames', exist_ok=True)
df_all.to_csv('data/justnames/consolidated_names_with_planet_candidate_nonplanet.csv', index=False)
