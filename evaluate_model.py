import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from helpers import load_dataset

X, y = load_dataset("data_students.txt")

model = RandomForestRegressor(
    n_estimators=50,
    min_samples_leaf=30,
    n_jobs=-1,
    random_state=42,
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("Running 5-fold cross-validation (trains 5 separate models, ~1 min total)...")
scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=1)

print()
print(f"R2 score per fold: {np.round(scores, 5)}")
print(f"Mean R2:  {scores.mean():.5f}")
print(f"Std dev:  {scores.std():.5f}")
print()
print("This mean (rounded down slightly for safety) is what we declared")
print("as 'Expected R2 Score' on the submission PDF.")
