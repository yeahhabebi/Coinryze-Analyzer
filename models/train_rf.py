# models/train_rf.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib, os

CSV = os.getenv("TRAIN_CSV", "frontend/coinryze_history.csv")
OUT = os.getenv("RF_OUT", "models/rf_model.joblib")
df = pd.read_csv(CSV)
X, y = [], []
win = 3
nums = df['number'].astype(int).tolist()
for i in range(win, len(nums)):
    seq = nums[i-win:i]
    feat = [seq[-1], sum(seq)/win, pd.Series(seq).std(ddof=0)]
    X.append(feat)
    y.append(nums[i])
if not X:
    raise SystemExit("Not enough data to train RF")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Train score:", clf.score(X_train, y_train))
print("Test score:", clf.score(X_test, y_test))
os.makedirs('models', exist_ok=True)
joblib.dump(clf, OUT)
print("Saved RF model to", OUT)
