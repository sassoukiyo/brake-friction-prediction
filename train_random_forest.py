
from sklearn.ensemble import RandomForestRegressor
from helpers import load_dataset, save_model

X, y = load_dataset("data_students.txt")

model = RandomForestRegressor(
    n_estimators=50,       # fewer trees -> smaller file, still averages out well
    min_samples_leaf=30,   # larger leaves -> smaller trees -> fits under ISIS's 20 MB upload cap
    n_jobs=-1,
    random_state=42,
)
print("Training...")
model.fit(X, y)
print(f"Done. Built {len(model.estimators_)} trees.")
save_model(model, "XX_000.pkl")
print("Saved to XX_000.pkl")
