import pandas as pd
import os 

TESS_DATASET_PATH = './TOI_2025.10.03_10.25.12.csv'
KEPLER_DATASET_PATH ='./cumulative_2025.10.03_10.24.49.csv'
K2_DATASET_PATH = './k2pandc_2025.10.03_10.25.19.csv'

if os.path.isdir('data/kepler') is False:
    
    # os.mkdir('tess')
    # os.mkdir('k2')
     
    kepler_df = pd.read_csv(KEPLER_DATASET_PATH, engine='python', on_bad_lines='skip')
    print(kepler_df.columns.to_list())
    
    # exoplanet archive disposition is the true labelfor the status of an exoplanet
    planet_df = kepler_df[kepler_df['koi_disposition'].isin(["CONFIRMED", "CANDIDATE"])]
    nonplanet_df = kepler_df[kepler_df['koi_disposition'].isin(['FALSE POSITIVE'])]
    
    os.makedirs('data/kepler/planet')
    os.makedirs('data/kepler/non_planet')
    
    planet_df.to_csv('data/kepler/planet/kepler_planets.csv')
    nonplanet_df.to_csv('data/kepler/non_planet/kepler_non_planet.csv')
    
if os.path.isdir('data/k2') is False:
    os.makedirs('data/k2/planet')
    os.makedirs('data/k2/non_planet')
    
    k2_df = pd.read_csv(K2_DATASET_PATH, engine='python', on_bad_lines='skip')
    
    # exoplanet archive disposition is the true labelfor the status of an exoplanet
    planet_df = k2_df[k2_df['disposition'].isin(['CONFIRMED', 'CANDIDATE'])]
    nonplanet_df = k2_df[k2_df['disposition'].isin(['FALSE POSITIVE', 'REFUTED'])]
    
    planet_df.to_csv('data/k2/planet/k2_planets.csv')
    nonplanet_df.to_csv('data/k2/non_planet/k2_non_planet.csv')
    
   
if os.path.isdir('data/tess') is False:
    os.makedirs('data/tess/planet')
    os.makedirs('data/tess/non_planet')
    
    tess_df = pd.read_csv(TESS_DATASET_PATH, engine='python', on_bad_lines='skip')
    print(tess_df.columns.to_list())
    
    # exoplanet archive disposition is the true labelfor the status of an exoplanet
    planet_df = tess_df[tess_df['tfopwg_disp'].isin(['CP','KP','PC'])]
    nonplanet_df = tess_df[tess_df['tfopwg_disp'].isin(['FP'])]
    
    planet_df.to_csv('data/tess/planet/tess_planets.csv')
    nonplanet_df.to_csv('data/tess/non_planet/tess_non_planet.csv')
    

