from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_DIR / "titanic.csv"

sns.set_theme(style="whitegrid")


def load_once():
    """
    Load Titanic from Seaborn exactly once and immediately save
    the offline fallback CSV.
    """
    df = sns.load_dataset("titanic")
    df.to_csv(CSV_PATH, index=False)
    return df


def profile_data(df):
    print("\n" + "=" * 70)
    print("DATASET PROFILE")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nInfo:")
    df.info()

    print("\nDescribe:")
    print(df.describe(include="all").T)

    print("\nMissing-value percentages:")
    missing = df.isna().mean().mul(100)
    missing = missing[missing > 0]

    for column, percentage in missing.items():
        print(f"{column}: {percentage:.2f}%")


def clean_data(df):
    """
    Apply the assignment's threshold rule.

    <5% missing:
        drop affected rows.

    5%-30%:
        impute.

    >30%:
        retain the column only when a defensible category/indicator
        is useful; otherwise drop it.
    """

    df = df.copy()

    missing = df.isna().mean() * 100

    print("\n" + "=" * 70)
    print("MISSING-VALUE DECISIONS")
    print("=" * 70)

    for column, percentage in missing[missing > 0].items():
        print(f"{column}: {percentage:.2f}%")

    # Under 5%: drop rows.
    # embarked is approximately 0.22% missing.
    if "embarked" in df.columns:
        rate = missing["embarked"]
        if rate < 5:
            df = df.dropna(subset=["embarked"])
            print(
                f"embarked: {rate:.2f}% missing -> "
                "drop rows because missingness is below 5%."
            )

    # Age is between 5% and 30%, so median imputation.
    if "age" in df.columns:
        rate = missing["age"]
        if 5 <= rate <= 30:
            median_age = df["age"].median()
            df["age"] = df["age"].fillna(median_age)
            print(
                f"age: {rate:.2f}% missing -> "
                f"median imputation ({median_age:.2f})."
            )

    # Deck has >30% missing. We encode missing as its own category
    # rather than attempting unreliable imputation.
    if "deck" in df.columns:
        rate = missing["deck"]
        if rate > 30:
            df["deck"] = df["deck"].astype("object").fillna("Missing")
            print(
                f"deck: {rate:.2f}% missing -> "
                "encode missing as its own category because "
                "imputation would be unreliable above 30%."
            )

    return df


def fare_outlier_analysis(df):
    print("\n" + "=" * 70)
    print("FARE AND AGE OUTLIERS")
    print("=" * 70)

    results = {}

    for column in ["age", "fare"]:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((df[column] < lower) | (df[column] > upper)).sum()

        results[column] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
            "count": int(count),
        }

        print(
            f"{column}: Q1={q1:.3f}, Q3={q3:.3f}, "
            f"IQR={iqr:.3f}, lower={lower:.3f}, "
            f"upper={upper:.3f}, outliers={count}"
        )

    return results


def univariate_analysis(df):
    print("\n" + "=" * 70)
    print("UNIVARIATE ANALYSIS")
    print("=" * 70)

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode().iloc[0]

    print(f"Fare mean: {fare_mean:.4f}")
    print(f"Fare median: {fare_median:.4f}")
    print(f"Fare mode: {fare_mode:.4f}")

    if fare_mean > fare_median > fare_mode:
        skew_text = "right-skewed"
    elif fare_mean < fare_median < fare_mode:
        skew_text = "left-skewed"
    else:
        skew_text = "approximately symmetric or mixed"

    print(
        f"Fare distribution: {skew_text}. "
        "This conclusion is based on the mean/median/mode ordering."
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.histplot(df["age"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Age Histogram")

    sns.boxplot(x=df["age"], ax=axes[0, 1])
    axes[0, 1].set_title("Age Box Plot")

    sns.histplot(df["fare"], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Fare Histogram")

    sns.boxplot(x=df["fare"], ax=axes[1, 1])
    axes[1, 1].set_title("Fare Box Plot")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_age_fare_univariate.png", dpi=150)
    plt.close()


def bivariate_analysis(df):
    print("\n" + "=" * 70)
    print("BIVARIATE SURVIVAL ANALYSIS")
    print("=" * 70)

    sex_rate = df.groupby("sex")["survived"].mean()
    pclass_rate = df.groupby("pclass")["survived"].mean()
    sex_class_rate = df.groupby(["sex", "pclass"])["survived"].mean()

    print("\nSurvival rate by sex:")
    print(sex_rate)

    print("\nSurvival rate by pclass:")
    print(pclass_rate)

    print("\nSurvival rate by sex and pclass:")
    print(sex_class_rate)

    # Explicit boolean masking examples.
    female_survivors = df[(df["sex"] == "female") & (df["survived"] == 1)]
    male_survivors = df[(df["sex"] == "male") & (df["survived"] == 1)]

    print(f"\nFemale survivors: {len(female_survivors)}")
    print(f"Male survivors: {len(male_survivors)}")

    numeric_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    corr = df[numeric_columns].corr()

    print("\nRequired 6x6 correlation matrix:")
    print(corr)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )
    plt.title("Titanic Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close()

    pairs = []

    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            a = numeric_columns[i]
            b = numeric_columns[j]
            pairs.append((a, b, corr.loc[a, b], abs(corr.loc[a, b])))

    pairs.sort(key=lambda x: x[3], reverse=True)

    print("\nTwo strongest absolute off-diagonal correlations:")
    for a, b, value, absolute in pairs[:2]:
        print(f"{a} vs {b}: correlation={value:.4f}")


def multivariate_story(df):
    print("\n" + "=" * 70)
    print("MULTIVARIATE DATA STORY")
    print("=" * 70)

    # 1
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="sex", y="survived", hue="sex", legend=False)
    plt.title("Survival Rate by Sex")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_survival_by_sex.png", dpi=150)
    plt.close()

    print(
        "Chart 1 interpretation: Survival differs substantially by sex. "
        "The female group has a higher observed survival proportion than "
        "the male group in this dataset."
    )

    # 2
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="pclass", y="survived", hue="pclass", legend=False)
    plt.title("Survival Rate by Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_survival_by_class.png", dpi=150)
    plt.close()

    print(
        "Chart 2 interpretation: Survival rate varies by passenger class. "
        "First-class passengers show a higher observed survival proportion "
        "than lower classes."
    )

    # 3
    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        hue="sex",
        errorbar=None,
    )
    plt.title("Survival Rate by Sex and Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_survival_sex_class.png", dpi=150)
    plt.close()

    print(
        "Chart 3 interpretation: Combining sex and class reveals that the "
        "survival difference associated with sex persists across classes, "
        "while passenger class also separates survival proportions."
    )

    # 4
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="pclass", y="fare", hue="sex")
    plt.title("Fare Distribution by Class and Sex")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_fare_class_sex.png", dpi=150)
    plt.close()

    print(
        "Chart 4 interpretation: Fare distributions differ strongly by "
        "passenger class. Higher fares are concentrated in higher classes, "
        "providing a financial proxy for the socioeconomic differences "
        "associated with class."
    )


def standardization_check(df):
    print("\n" + "=" * 70)
    print("EXPLORATORY STANDARDIZATION CHECK")
    print("=" * 70)

    scaler = StandardScaler()

    standardized = scaler.fit_transform(df[["age", "fare"]])

    standardized_df = pd.DataFrame(
        standardized,
        columns=["age_z", "fare_z"],
        index=df.index,
    )

    print("\nBefore:")
    print(df[["age", "fare"]].agg(["mean", "std"]))

    print("\nAfter:")
    print(standardized_df.agg(["mean", "std"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(standardized_df["age_z"], kde=True, ax=axes[0])
    axes[0].set_title("Standardized Age")

    sns.histplot(standardized_df["fare_z"], kde=True, ax=axes[1])
    axes[1].set_title("Standardized Fare")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_standardization_check.png", dpi=150)
    plt.close()


def main():
    print("=" * 70)
    print("TITANIC EDA PIPELINE")
    print("=" * 70)

    # The ONLY network/cache load of the raw Titanic dataset.
    df = load_once()

    print(f"\nOffline fallback saved to: {CSV_PATH}")

    profile_data(df)

    clean_df = clean_data(df)

    print(f"\nCleaned shape: {clean_df.shape}")

    # Keep a separate cleaned CSV for reproducibility.
    clean_df.to_csv(BASE_DIR / "titanic_cleaned.csv", index=False)

    fare_outlier_analysis(clean_df)
    univariate_analysis(clean_df)
    bivariate_analysis(clean_df)
    multivariate_story(clean_df)
    standardization_check(clean_df)

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)
    print(f"Raw offline fallback: {CSV_PATH}")
    print(f"Cleaned dataset: {BASE_DIR / 'titanic_cleaned.csv'}")
    print(f"Charts: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

