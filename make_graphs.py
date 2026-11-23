import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = "team-07-modeling/models/"
df = pd.read_csv(M + "xgboost_test_forecast.csv", parse_dates=["DATE"])

# 1. Network total: actual vs naive vs forecast
net = df.groupby("DATE")[["OD_PAX", "naive", "forecast"]].sum()
net.plot(figsize=(10, 5), title="Network passengers: actual vs forecast")
plt.ylabel("OD_PAX"); plt.tight_layout(); plt.savefig("network_forecast.png", dpi=200); plt.close()

# 2. WAPE by horizon
g = df.assign(m=(df.forecast - df.OD_PAX).abs(), n=(df.naive - df.OD_PAX).abs()).groupby("h")[["m", "n", "OD_PAX"]].sum()
pd.DataFrame({"model": g.m / g.OD_PAX, "naive": g.n / g.OD_PAX}).plot(marker="o", figsize=(10, 5), title="WAPE by horizon (months ahead)")
plt.ylabel("WAPE"); plt.tight_layout(); plt.savefig("wape_by_horizon.png", dpi=200); plt.close()

# 3. 80% interval coverage by horizon
cov = ((df.OD_PAX >= df.low) & (df.OD_PAX <= df.high)).groupby(df.h).mean()
cov.plot(marker="o", figsize=(10, 5), title="80% interval coverage by horizon")
plt.axhline(0.8, color="red", linestyle="--"); plt.ylabel("coverage")
plt.tight_layout(); plt.savefig("coverage_by_horizon.png", dpi=200); plt.close()

# 4. Future 24-month forecast
try:
    fut = pd.read_csv(glob.glob(M + "xgboost_forecast_*_to_*.csv")[0], parse_dates=["DATE"])
    fut.groupby("DATE")["forecast"].sum().plot(figsize=(10, 5), title="Future 24-month network forecast")
    plt.ylabel("OD_PAX"); plt.tight_layout(); plt.savefig("future_forecast.png", dpi=200); plt.close()
except Exception as e:
    print("skipped future forecast:", e)

# 5. Feature importance, read from results.txt
try:
    lines = open("results.txt").read().splitlines()
    i = lines.index("feature importance (gain):")
    vals = {}
    for ln in lines[i + 1:]:
        if not ln.strip():
            break
        k, v = ln.rsplit(None, 1)
        vals[k] = float(v)
    pd.Series(vals).head(15)[::-1].plot.barh(figsize=(8, 6), title="Top 15 features (gain)")
    plt.tight_layout(); plt.savefig("importance.png", dpi=200); plt.close()
except Exception as e:
    print("skipped importance:", e)

print("done")
