import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# =========================
# 1. FAIRNESS METRICS
# =========================

def fairness_metrics(y_true, y_pred, protected):
    df = pd.DataFrame({
        "y_true": pd.Series(y_true).reset_index(drop=True),
        "y_pred": pd.Series(y_pred).reset_index(drop=True),
        "protected": pd.Series(protected).reset_index(drop=True)
    })

    # assume encoded protected groups are 0 and 1
    group0 = df[df["protected"] == 0]
    group1 = df[df["protected"] == 1]

    if len(group0) == 0 or len(group1) == 0:
        return np.nan, np.nan, np.nan

    # Statistical Parity Difference
    spd = group1["y_pred"].mean() - group0["y_pred"].mean()

    # Disparate Impact
    di = group1["y_pred"].mean() / (group0["y_pred"].mean() + 1e-6)

    # Equal Opportunity Difference
    group0_pos = group0[group0["y_true"] == 1]
    group1_pos = group1[group1["y_true"] == 1]

    tpr0 = group0_pos["y_pred"].mean() if len(group0_pos) > 0 else 0
    tpr1 = group1_pos["y_pred"].mean() if len(group1_pos) > 0 else 0
    eod = tpr1 - tpr0

    return spd, di, eod


# =========================
# 2. ENCODING FUNCTION
# =========================

def encode_dataframe(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


# =========================
# 3. MODEL FUNCTION
# =========================

def run_models(X_train, X_test, y_train, y_test, dataset_name, protected_col):
    results = []

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        spd, di, eod = fairness_metrics(y_test, y_pred, X_test[protected_col])

        print(f"\n===== {dataset_name} | {name} =====")
        print("Accuracy:", round(acc, 4))
        print("SPD:", round(spd, 4) if pd.notna(spd) else "NA")
        print("DI:", round(di, 4) if pd.notna(di) else "NA")
        print("EOD:", round(eod, 4) if pd.notna(eod) else "NA")
        print(classification_report(y_test, y_pred))

        results.append({
            "Dataset": dataset_name,
            "Model": name,
            "Accuracy": round(acc, 4),
            "SPD": round(spd, 4) if pd.notna(spd) else np.nan,
            "DI": round(di, 4) if pd.notna(di) else np.nan,
            "EOD": round(eod, 4) if pd.notna(eod) else np.nan
        })

    return pd.DataFrame(results)


# =========================
# 4. LOAD ADULT DATASET
# =========================

adult_columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

adult_df = pd.read_csv(
    "adult.data",
    names=adult_columns,
    skipinitialspace=True
)

adult_df = adult_df.replace("?", np.nan).dropna()
adult_df.columns = adult_df.columns.str.strip().str.lower().str.replace(" ", "_")
adult_df["income"] = adult_df["income"].astype(str).str.strip()

print("Adult Dataset Preview:")
print(adult_df.head())
print("\nAdult Dataset Columns:")
print(adult_df.columns.tolist())


# =========================
# 5. LOAD AND MERGE HR DATA
# =========================

hr_core = pd.read_excel("HRDatabase.xlsx", engine="openpyxl")
hr_emp = pd.read_excel("EmployeeInformation.xlsx", engine="openpyxl")
hr_dept = pd.read_excel("DepartmentInformation.xlsx", engine="openpyxl")

for df in [hr_core, hr_emp, hr_dept]:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\nHR Core Columns:")
print(hr_core.columns.tolist())

print("\nEmployee Info Columns:")
print(hr_emp.columns.tolist())

print("\nDepartment Info Columns:")
print(hr_dept.columns.tolist())

# merge carefully
hr_df = hr_core.merge(hr_emp, on="id", how="left")
hr_df = hr_df.merge(hr_dept, on="department", how="left")

print("\nMerged HR Dataset Preview:")
print(hr_df.head())
print("\nMerged HR Dataset Columns:")
print(hr_df.columns.tolist())
print("\nMerged HR dtypes:")
print(hr_df.dtypes)


# =========================
# 6. CREATE HR TARGET
# =========================
# 1 = active employee
# 0 = terminated employee

if "termination_date" in hr_df.columns:
    hr_df["target"] = hr_df["termination_date"].isna().astype(int)
else:
    raise ValueError("termination_date column not found in merged HR dataset.")

print("\nHR Target Distribution:")
print(hr_df["target"].value_counts(dropna=False))


# =========================
# 7. CLEAN HR DATA
# =========================

# ensure gender exists
if "gender" not in hr_df.columns:
    np.random.seed(42)
    hr_df["gender"] = np.random.choice([0, 1], size=len(hr_df))

# drop leakage / IDs / raw dates
drop_cols = [
    "id",
    "employee",
    "birth_date",
    "hire_date",
    "termination_date",
    "termination_reason",
    "last_promotion_date"
]

drop_cols = [col for col in drop_cols if col in hr_df.columns]
hr_df = hr_df.drop(columns=drop_cols)

# auto-drop any remaining datetime columns
datetime_cols = hr_df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
if datetime_cols:
    print("\nDropping datetime columns:", datetime_cols)
    hr_df = hr_df.drop(columns=datetime_cols)

print("\nHR Columns After Cleanup:")
print(hr_df.columns.tolist())
print("\nHR dtypes After Cleanup:")
print(hr_df.dtypes)


# =========================
# 8. ENCODE DATA
# =========================

adult_encoded = encode_dataframe(adult_df)
hr_encoded = encode_dataframe(hr_df)

print("\nEncoded HR dtypes:")
print(hr_encoded.dtypes)


# =========================
# 9. PREPARE ADULT FEATURES
# =========================

adult_target = "income"
adult_protected = "sex"

X_adult = adult_encoded.drop(columns=[adult_target])
y_adult = adult_encoded[adult_target]

X_adult_train, X_adult_test, y_adult_train, y_adult_test = train_test_split(
    X_adult, y_adult, test_size=0.2, random_state=42, stratify=y_adult
)


# =========================
# 10. PREPARE HR FEATURES
# =========================

hr_target = "target"
hr_protected = "gender"

X_hr = hr_encoded.drop(columns=[hr_target])
y_hr = hr_encoded[hr_target]

print("\nAdult Target Distribution:")
print(y_adult.value_counts())

print("\nHR Target Distribution:")
print(y_hr.value_counts())

X_hr_train, X_hr_test, y_hr_train, y_hr_test = train_test_split(
    X_hr, y_hr, test_size=0.2, random_state=42, stratify=y_hr
)


# =========================
# 11. RUN MODELS
# =========================

adult_results = run_models(
    X_adult_train, X_adult_test, y_adult_train, y_adult_test,
    "Adult Dataset", adult_protected
)

hr_results = run_models(
    X_hr_train, X_hr_test, y_hr_train, y_hr_test,
    "Merged HR Dataset", hr_protected
)

all_results = pd.concat([adult_results, hr_results], ignore_index=True)

print("\n===== FINAL RESULTS =====")
print(all_results)


# =========================
# 12. SAVE RESULTS
# =========================

all_results.to_csv("baseline_results_with_fairness.csv", index=False)
print("\nSaved results to baseline_results_with_fairness.csv")


# =========================
# 13. PLOT ACCURACY
# =========================

for dataset_name in all_results["Dataset"].unique():
    subset = all_results[all_results["Dataset"] == dataset_name]

    plt.figure(figsize=(8, 5))
    plt.bar(subset["Model"], subset["Accuracy"])
    plt.title(f"{dataset_name} Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_accuracy.png")
    plt.show()


# =========================
# 14. PLOT SPD
# =========================

for dataset_name in all_results["Dataset"].unique():
    subset = all_results[all_results["Dataset"] == dataset_name]

    plt.figure(figsize=(8, 5))
    plt.bar(subset["Model"], subset["SPD"])
    plt.title(f"{dataset_name} Statistical Parity Difference")
    plt.xlabel("Model")
    plt.ylabel("SPD")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_spd.png")
    plt.show()


# =========================
# 15. PLOT DI
# =========================

for dataset_name in all_results["Dataset"].unique():
    subset = all_results[all_results["Dataset"] == dataset_name]

    plt.figure(figsize=(8, 5))
    plt.bar(subset["Model"], subset["DI"])
    plt.title(f"{dataset_name} Disparate Impact")
    plt.xlabel("Model")
    plt.ylabel("DI")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_di.png")
    plt.show()


# =========================
# 16. PLOT EOD
# =========================

for dataset_name in all_results["Dataset"].unique():
    subset = all_results[all_results["Dataset"] == dataset_name]

    plt.figure(figsize=(8, 5))
    plt.bar(subset["Model"], subset["EOD"])
    plt.title(f"{dataset_name} Equal Opportunity Difference")
    plt.xlabel("Model")
    plt.ylabel("EOD")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_eod.png")
    plt.show()

print("\nAll graphs saved successfully.")