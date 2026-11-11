import pandas as pd
import numpy as np

print("==================================================")
print("                     EDA                          ")
print("==================================================")

#Structural inspection
data_path = "AA_PAX_FORECASTING.csv"
df = pd.read_csv(data_path)

print(f"\n Dataset Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")

print("\n Column Data Types & Memory Usage:")
print(df.info())

print("\n Missing Values Audit:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("-> No missing values detected across any columns.")
else:
    print(missing[missing > 0])

print("\n Basic Statistical Summary (Numerical Features):")
print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])


#Target Variable Analysis (OD_REV_USD)
print("\n--------------------------------------------------")
print("       Target Variable Analysis (OD_REV_USD)       ")
print("--------------------------------------------------")
target = 'OD_REV_USD'
if target in df.columns:
    rev_mean = df[target].mean()
    rev_median = df[target].median()
    rev_std = df[target].std()
    rev_skew = df[target].skew()
    
    print(f"-> Mean Revenue: ${rev_mean:,.2f}")
    print(f"-> Median Revenue: ${rev_median:,.2f}")
    print(f"-> Standard Deviation: ${rev_std:,.2f}")
    print(f"-> Skewness: {rev_skew:.4f} (High positive skew indicates long tail of high-value routes/passengers)")
    
    #Quantiles to check distribution spread
    quantiles = df[target].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print("\nRevenue Quantiles:")
    print(quantiles)
else:
    print(f"Warning: Target column '{target}' not found.")


#Operational & Network Analysis (AIRPORTS & ROUTES)
print("\n--------------------------------------------------")
print(" Operational & Network Analysis")
print("--------------------------------------------------")
if 'ORG_AIRPRT' in df.columns and 'DES_AIRPRT' in df.columns:
    print(f"-> Unique Origin Airports: {df['ORG_AIRPRT'].nunique()}")
    print(f"-> Unique Destination Airports: {df['DES_AIRPRT'].nunique()}")
    
    #Create route identifier
    df['ROUTE'] = df['ORG_AIRPRT'] + ' -> ' + df['DES_AIRPRT']
    top_routes = df['ROUTE'].value_counts().head(10)
    print("\nTop 10 Most Frequent Routes by Record Count:")
    print(top_routes)
    
    #Top revenue-generating routes
    top_rev_routes = df.groupby('ROUTE')['OD_REV_USD'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 Highest Grossing Routes (Total Revenue USD):")
    print(top_rev_routes)


#Passenger volume & Yield analysis (OD_PAX)
print("\n--------------------------------------------------")
print(" Passenger volume & Yield analysis (OD_PAX)")
print("--------------------------------------------------")
if 'OD_PAX' in df.columns and 'OD_REV_USD' in df.columns:
    total_pax = df['OD_PAX'].sum()
    mean_pax = df['OD_PAX'].mean()
    print(f"-> Total Passenger Volume Across Dataset: {total_pax:,.0f}")
    print(f"-> Average Passengers per Record: {mean_pax:.2f}")
    
    #Calculate yield (Revenue per passenger) --> avoiding division by zero
    df['YIELD_PER_PAX'] = np.where(df['OD_PAX'] > 0, df['OD_REV_USD'] / df['OD_PAX'], 0)
    mean_yield = df['YIELD_PER_PAX'].mean()
    print(f"-> Average Yield (Revenue / Passenger): ${mean_yield:.2f} per pax")


#Seasonal patterns
print("\n--------------------------------------------------")
print("Seasonal trends")
print("--------------------------------------------------")
if 'DEP_YEAR' in df.columns and 'DEP_MONTH' in df.columns:
    print("-> Recorded Departure Years:", sorted(df['DEP_YEAR'].unique()))
    
    #Revenue aggregation by month across years
    monthly_rev = df.groupby(['DEP_YEAR', 'DEP_MONTH'])['OD_REV_USD'].agg(['sum', 'mean', 'count']).reset_index()
    print("\nMonthly Revenue Performance Overview (Sample):")
    print(monthly_rev.head(12))
    
    #Seasonality check: Average revenue by month
    seasonality = df.groupby('DEP_MONTH')['OD_REV_USD'].mean()
    print("\nAverage Revenue by Calendar Month (Seasonality Check):")
    print(seasonality)
    
    #Route-level unit economics
    route_summary = df.groupby(['ORG_AIRPRT', 'DES_AIRPRT']).agg(
        TOTAL_REV=('OD_REV_USD', 'sum'),
        TOTAL_PAX=('OD_PAX', 'sum'),
        RECORD_COUNT=('OD_REV_USD', 'count')
    ).reset_index()

    route_summary['AVG_YIELD_PER_PAX'] = route_summary['TOTAL_REV'] / route_summary['TOTAL_PAX']

    print("\n--- Top 10 Highest Yield Routes (Revenue per Passenger) ---")
    print(route_summary.sort_values(by='AVG_YIELD_PER_PAX', ascending=False).head(10))
    
    #Yearly macro performance
    yearly_trend = df.groupby('DEP_YEAR').agg(
        TOTAL_REVENUE=('OD_REV_USD', 'sum'),
        TOTAL_PASSENGERS=('OD_PAX', 'sum'),
        AVG_REV_PER_RECORD=('OD_REV_USD', 'mean'),
        RECORD_COUNT=('OD_REV_USD', 'count')
    ).reset_index()

    print("\n--- Year-over-Year Macro Trends ---")
    print(yearly_trend)
    
    #Target distribution transformation check
    df['LOG_REV'] = np.log1p(df['OD_REV_USD'])
    print(f"\n--- Log-Transformed Revenue Skewness: {df['LOG_REV'].skew():.4f} ---")
    print("Log transformation significantly reduces skew, moving it closer to a normal distribution.")
    
    #Numerical correlation matrix
    print("\n--- Feature Correlation Matrix ---")
    print(df[['DEP_MONTH', 'DEP_YEAR', 'OD_PAX', 'OD_REV_USD']].corr())
    
    #Top origin airports by total revenue
    top_orgs = df.groupby('ORG_AIRPRT')['OD_REV_USD'].sum().sort_values(ascending=False).head(10)
    print("\n--- Top 10 Origin Airports by Revenue ---")
    print(top_orgs)
    
    #Pivot table of revenue by year and month
    seasonality_pivot = df.pivot_table(index='DEP_YEAR', columns='DEP_MONTH', values='OD_REV_USD', aggfunc='sum')
    print("\n--- YoY Monthly Revenue Seasonality Matrix ---")
    print(seasonality_pivot)

print("\n==================================================")
print("                EDA Complete                       ")
print("==================================================")

