import os
import pandas as pd
import numpy as np

print("==================================================")
print("       AMERICAN AIRLINES: FINAL DEEP-DIVE EDA     ")
print("==================================================")

# Helper to find files whether run from root or inside team-05-modeling/
def resolve_path(filename):
    if os.path.exists(filename):
        return filename
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return parent_path
    raise FileNotFoundError(f"Could not find '{filename}' in current or parent directory.")

# --------------------------------------------------
# [1.0] STRUCTURAL INSPECTION & CORE DATASET
# --------------------------------------------------
df = pd.read_csv(resolve_path("AA_PAX_FORECASTING.csv"))
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print(f"\n[1.1] Dataset Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\n[1.2] Column Data Types & Memory Usage:")
print(df.info())

print("\n[1.3] Missing Values Audit:")
missing = df.isnull().sum()
print("-> No missing values detected." if missing.sum() == 0 else missing[missing > 0])

print("\n[1.4] Basic Statistical Summary (Numerical Features):")
print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])

# --------------------------------------------------
# [2.0] TARGET VARIABLE ANALYSIS (OD_REV_USD)
# --------------------------------------------------
print("\n--------------------------------------------------")
print("      [2.0] Target Variable Analysis (OD_REV_USD) ")
print("--------------------------------------------------")
target = 'OD_REV_USD'
print(f"-> Mean Revenue:       ${df[target].mean():,.2f}")
print(f"-> Median Revenue:     ${df[target].median():,.2f}")
print(f"-> Standard Deviation: ${df[target].std():,.2f}")
print(f"-> Raw Skewness:       {df[target].skew():.4f}")

df['LOG_REV'] = np.log1p(df[target])
print(f"-> Log1p Skewness:     {df['LOG_REV'].skew():.4f} (Normalizes long tail for modeling)")
print("\nRevenue Quantiles:")
print(df[target].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

# --------------------------------------------------
# [3.0] OPERATIONAL, NETWORK & UNIT ECONOMICS
# --------------------------------------------------
print("\n--------------------------------------------------")
print("    [3.0] Operational, Network & Unit Economics   ")
print("--------------------------------------------------")
df['ROUTE'] = df['ORG_AIRPRT'] + ' -> ' + df['DES_AIRPRT']
df['YIELD_PER_PAX'] = np.where(df['OD_PAX'] > 0, df['OD_REV_USD'] / df['OD_PAX'], 0)

print(f"-> Unique Origin Airports:      {df['ORG_AIRPRT'].nunique()}")
print(f"-> Unique Destination Airports: {df['DES_AIRPRT'].nunique()}")
print(f"-> Total Passenger Volume:      {df['OD_PAX'].sum():,.0f}")
print(f"-> Average Yield per Passenger: ${df['YIELD_PER_PAX'].mean():.2f}")

print("\nTop 10 Highest Grossing Routes (Total Revenue USD):")
print(df.groupby('ROUTE')['OD_REV_USD'].sum().sort_values(ascending=False).head(10))

route_summary = df.groupby(['ORG_AIRPRT', 'DES_AIRPRT']).agg(
    TOTAL_REV=('OD_REV_USD', 'sum'),
    TOTAL_PAX=('OD_PAX', 'sum'),
    RECORD_COUNT=('OD_REV_USD', 'count')
).reset_index()
route_summary['AVG_YIELD_PER_PAX'] = route_summary['TOTAL_REV'] / route_summary['TOTAL_PAX']
print("\nTop 10 Highest Yield Routes (Revenue per Passenger):")
print(route_summary.sort_values(by='AVG_YIELD_PER_PAX', ascending=False).head(10))

# --------------------------------------------------
# [4.0] TEMPORAL & SEASONALITY PATTERNS
# --------------------------------------------------
print("\n--------------------------------------------------")
print("       [4.0] Temporal & Seasonality Patterns      ")
print("--------------------------------------------------")
yearly_trend = df.groupby('DEP_YEAR').agg(
    TOTAL_REVENUE=('OD_REV_USD', 'sum'),
    TOTAL_PASSENGERS=('OD_PAX', 'sum'),
    AVG_REV_PER_RECORD=('OD_REV_USD', 'mean'),
    RECORD_COUNT=('OD_REV_USD', 'count')
).reset_index()
print("\nYear-over-Year Macro Trends:")
print(yearly_trend)

print("\nAverage Revenue by Calendar Month (Seasonality Check):")
print(df.groupby('DEP_MONTH')['OD_REV_USD'].mean())

# --------------------------------------------------
# [5.0] INSPECT & MERGE AA_AIRPORT_INFO.CSV
# --------------------------------------------------
print("\n--------------------------------------------------")
print("   [5.0] Airport Metadata Enrichment (AIRPORT_INFO)")
print("--------------------------------------------------")
airport_info = pd.read_csv(resolve_path("AA_AIRPORT_INFO.csv"))
print("AA_AIRPORT_INFO.csv Columns:", list(airport_info.columns))
print(airport_info.head(3))

# Auto-detect the IATA/Airport code column in AA_AIRPORT_INFO.csv
code_candidates = [c for c in airport_info.columns if any(k in c.upper() for k in ['AIRPRT', 'IATA', 'LOCID', 'CODE', 'CD'])]
airport_key = code_candidates[0] if code_candidates else airport_info.columns[0]
airport_info[airport_key] = airport_info[airport_key].astype(str).str.strip().str.upper()
airport_info_dedup = airport_info.drop_duplicates(subset=[airport_key])

# Merge Origin & Destination Airport Info
org_info = airport_info_dedup.add_prefix("ORG_")
des_info = airport_info_dedup.add_prefix("DES_")

df = df.merge(org_info, left_on="ORG_AIRPRT", right_on=f"ORG_{airport_key}", how="left")
df = df.merge(des_info, left_on="DES_AIRPRT", right_on=f"DES_{airport_key}", how="left")
print(f"-> Merged AA_AIRPORT_INFO using key '{airport_key}'.")

# --------------------------------------------------
# [6.0] MERGE FAA ENPLANEMENTS & CARGO DATA (2025)
# --------------------------------------------------
print("\n--------------------------------------------------")
print("   [6.0] External FAA Activity Data (2025)        ")
print("--------------------------------------------------")
enp = pd.read_excel(resolve_path("arp-cy2025-commercial-service-enplanements.xlsx"))
enp['Locid'] = enp['Locid'].astype(str).str.strip().str.upper()
enp_clean = enp[['Locid', 'Hub', 'CY 25 Enplanements', '% Change']].drop_duplicates('Locid')

cargo = pd.read_excel(resolve_path("arp-CY2025-all-cargo-airports.xlsx"))
cargo['Locid'] = cargo['Locid'].astype(str).str.strip().str.upper()
cargo_clean = cargo[['Locid', '2025 Landed\nWeight (lbs.)', '% Change']].drop_duplicates('Locid')

# Join Enplanements (Origin & Destination)
df = df.merge(enp_clean, left_on='ORG_AIRPRT', right_on='Locid', how='left').drop(columns=['Locid'])
df = df.rename(columns={
    'Hub': 'ORG_FAA_HUB',
    'CY 25 Enplanements': 'ORG_ENPLANEMENTS_25',
    '% Change': 'ORG_ENP_YOY_CHG'
})

df = df.merge(enp_clean, left_on='DES_AIRPRT', right_on='Locid', how='left').drop(columns=['Locid'])
df = df.rename(columns={
    'Hub': 'DES_FAA_HUB',
    'CY 25 Enplanements': 'DES_ENPLANEMENTS_25',
    '% Change': 'DES_ENP_YOY_CHG'
})

# Join Cargo Weight (Origin & Destination)
df = df.merge(cargo_clean, left_on='ORG_AIRPRT', right_on='Locid', how='left').drop(columns=['Locid'])
df = df.rename(columns={
    '2025 Landed\nWeight (lbs.)': 'ORG_CARGO_LBS_25',
    '% Change': 'ORG_CARGO_YOY_CHG'
})

df = df.merge(cargo_clean, left_on='DES_AIRPRT', right_on='Locid', how='left').drop(columns=['Locid'])
df = df.rename(columns={
    '2025 Landed\nWeight (lbs.)': 'DES_CARGO_LBS_25',
    '% Change': 'DES_CARGO_YOY_CHG'
})

# Fill missing hub tiers as 'INTL_OR_NON_FAA' (e.g., LHR, NRT, CUN)
df['ORG_FAA_HUB'] = df['ORG_FAA_HUB'].fillna('INTL_OR_OTHER')
df['DES_FAA_HUB'] = df['DES_FAA_HUB'].fillna('INTL_OR_OTHER')

print("\nRevenue & Yield by Origin FAA Hub Tier (L=Large, M=Medium, S=Small, INTL=International):")
print(df.groupby('ORG_FAA_HUB').agg(
    RECORDS=('OD_REV_USD', 'count'),
    TOTAL_REV=('OD_REV_USD', 'sum'),
    AVG_REV=('OD_REV_USD', 'mean'),
    AVG_PAX=('OD_PAX', 'mean'),
    AVG_YIELD=('YIELD_PER_PAX', 'mean')
))

# --------------------------------------------------
# [7.0] MERGE CENSUS METRO POPULATION & MIGRATION (2025)
# --------------------------------------------------
print("\n--------------------------------------------------")
print("   [7.0] External Census Metro Demographics (2025)")
print("--------------------------------------------------")
cbsa = pd.read_csv(resolve_path("cbsa-est2025-alldata.csv"), encoding="latin1")
metro = cbsa[cbsa['LSAD'] == 'Metropolitan Statistical Area'][
    ['CBSA', 'NAME', 'POPESTIMATE2025', 'NPOPCHG2025', 'NETMIG2025', 'INTERNATIONALMIG2025', 'DOMESTICMIG2025']
].copy()

# Extract primary city & state from Census 'NAME' (e.g., "Dallas-Fort Worth-Arlington, TX" -> "DALLAS", "TX")
metro['PRIMARY_CITY'] = metro['NAME'].str.split(',').str[0].str.split('-').str[0].str.strip().str.upper()
metro['STATE_CD'] = metro['NAME'].str.split(',').str[1].str.strip().str.upper()
metro_city_lookup = metro.sort_values('POPESTIMATE2025', ascending=False).drop_duplicates(['PRIMARY_CITY', 'STATE_CD'])

# Use FAA Enplanements City + State to bridge Airport Code -> Census Metro
enp_geo = enp[['Locid', 'City', 'ST']].drop_duplicates('Locid').copy()
enp_geo['CITY_CLEAN'] = enp_geo['City'].astype(str).str.split('/').str[0].str.strip().str.upper()
enp_geo['ST_CLEAN'] = enp_geo['ST'].astype(str).str.strip().str.upper()

airport_metro = enp_geo.merge(
    metro_city_lookup,
    left_on=['CITY_CLEAN', 'ST_CLEAN'],
    right_on=['PRIMARY_CITY', 'STATE_CD'],
    how='left'
)[['Locid', 'NAME', 'POPESTIMATE2025', 'NPOPCHG2025', 'NETMIG2025', 'INTERNATIONALMIG2025']].rename(columns={
    'NAME': 'METRO_NAME',
    'POPESTIMATE2025': 'METRO_POP_25',
    'NPOPCHG2025': 'METRO_POP_CHG_25',
    'NETMIG2025': 'METRO_NET_MIG_25',
    'INTERNATIONALMIG2025': 'METRO_INTL_MIG_25'
})

# Merge Origin & Destination Metro Demographics
df = df.merge(airport_metro.add_prefix('ORG_'), left_on='ORG_AIRPRT', right_on='ORG_Locid', how='left').drop(columns=['ORG_Locid'])
df = df.merge(airport_metro.add_prefix('DES_'), left_on='DES_AIRPRT', right_on='DES_Locid', how='left').drop(columns=['DES_Locid'])

metro_match_rate = df['ORG_METRO_POP_25'].notnull().mean() * 100
print(f"-> Origin Census Metro Match Rate: {metro_match_rate:.1f}%")

# --------------------------------------------------
# [8.0] FINAL CORRELATION MATRIX (CORE + EXTERNAL)
# --------------------------------------------------
print("\n--------------------------------------------------")
print("   [8.0] Combined Feature Correlation Matrix       ")
print("--------------------------------------------------")
corr_features = [
    'OD_REV_USD', 'LOG_REV', 'OD_PAX', 'DEP_MONTH', 'DEP_YEAR',
    'ORG_ENPLANEMENTS_25', 'DES_ENPLANEMENTS_25',
    'ORG_CARGO_LBS_25', 'DES_CARGO_LBS_25',
    'ORG_METRO_POP_25', 'DES_METRO_POP_25',
    'ORG_METRO_NET_MIG_25', 'DES_METRO_NET_MIG_25'
]
available_corr = [c for c in corr_features if c in df.columns]
print(df[available_corr].corr()['OD_REV_USD'].sort_values(ascending=False))

print("\n==================================================")
print("             FINAL ENRICHED EDA COMPLETE          ")
print("==================================================")