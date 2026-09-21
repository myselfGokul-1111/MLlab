import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from PIL import Image

print("Imported successfully")


def eda_tabular(df, target=None):
    """Performs exploratory data analysis on a tabular DataFrame, including

    categorical feature distributions and target class analysis.
    """
    print("Shape:", df.shape)
    print("\nColumn types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # Helper function to plot class distributions
    def _plot_distribution(col_name, palette="viridis"):
        plt.figure(figsize=(7, 4.5))
        counts = df[col_name].value_counts(dropna=False).sort_index()
        total = len(df[col_name].dropna())

        ax = sns.barplot(
            x=counts.index.astype(str),
            y=counts.values,
            palette=palette,
            hue=counts.index.astype(str),
            legend=False,
        )

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                percentage = f"{(height / total) * 100:.1f}%"
                ax.annotate(
                    f"{int(height)}\n({percentage})",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords="offset points",
                )

        plt.title(f"Class Distribution: '{col_name}'", fontsize=12, pad=12)
        plt.xlabel(col_name, fontsize=10)
        plt.ylabel("Count", fontsize=10)

        # Rotate labels if categories are long strings or numerous
        if len(counts) > 5 or counts.index.astype(str).str.len().max() > 8:
            plt.xticks(rotation=45, ha="right")

        plt.ylim(0, counts.max() * 1.18)
        plt.tight_layout()
        plt.show()

    # 1. Numeric Feature Distributions
    if len(numeric_cols) > 0:
        print("\nSummary statistics:\n", df.describe())
        df[numeric_cols].hist(figsize=(14, 10), bins=20)
        plt.suptitle("Numeric Feature Distributions", y=1.02)
        plt.tight_layout()
        plt.show()

    # 2. Correlation Heatmap
    if len(numeric_cols) > 1:
        fig_size = max(8, len(numeric_cols) * 0.6)
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        show_annot = len(numeric_cols) <= 12

        sns.heatmap(
            df[numeric_cols].corr(),
            annot=show_annot,
            fmt=".2f",
            annot_kws={"size": 8},
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar=True,
        )

        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    # 3. Categorical Distributions (If no target is specified, plot all categorical features)
    if target is None and len(categorical_cols) > 0:
        print("\nPlotting Categorical Distributions...")
        for col in categorical_cols:
            _plot_distribution(col, palette="Set2")

    # 4. Target Class Distribution (Explicitly provided target)
    if target is not None:
        if target in df.columns:
            _plot_distribution(target, palette="viridis")
        else:
            print(f"\nWarning: Target column '{target}' not found in DataFrame.")

    # 5. Message length distribution (for NLP columns)
    if "message" in df.columns:
        df["message_length"] = df["message"].astype(str).str.len()
        plt.figure(figsize=(8, 4))
        plt.hist(df["message_length"], bins=30, color="skyblue", edgecolor="black")
        plt.title("Message Length Distribution")
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    return numeric_cols, categorical_cols


def eda_images(df, label_col="label", image_size=(28, 28), samples=9):
    """Performs exploratory data analysis on a pixel-based image DataFrame."""
    print("\nShape:", df.shape)
    print("\nFirst 5 Rows:\n", df.head())
    print("\nData Types:\n", df.dtypes.value_counts())
    print("\nMissing Values:", df.isnull().sum().sum())
    print("Duplicate Rows:", df.duplicated().sum())

    labels = df[label_col]
    pixels = df.drop(columns=[label_col])

    print("\nNumber of Images:", len(df))
    print("Image Size:", image_size)
    print("Number of Classes:", labels.nunique())

    # Class Distribution
    plt.figure(figsize=(8, 4))
    labels.value_counts().sort_index().plot(kind="bar", color="indigo")
    plt.title("Class Distribution")
    plt.xlabel("Class / Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Sample Images Grid
    plt.figure(figsize=(8, 8))
    indices = np.random.choice(len(df), samples, replace=False)
    grid = int(np.ceil(np.sqrt(samples)))

    for i, idx in enumerate(indices):
        plt.subplot(grid, grid, i + 1)
        img = pixels.iloc[idx].values.reshape(image_size)
        plt.imshow(img, cmap="gray")
        plt.title(f"Label: {labels.iloc[idx]}")
        plt.axis("off")

    plt.suptitle("Sample Images", y=0.95)
    plt.tight_layout()
    plt.show()

    # Pixel Intensity Distribution
    plt.figure(figsize=(8, 4))
    plt.hist(pixels.values.ravel(), bins=50, color="gray", edgecolor="black")
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Average Image
    avg_img = pixels.mean(axis=0).values.reshape(image_size)

    plt.figure(figsize=(4, 4))
    plt.imshow(avg_img, cmap="gray")
    plt.title("Average Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Pixel Statistics
    print("\nPixel Statistics")
    print("-------------------------")
    print("Minimum Pixel :", pixels.values.min())
    print("Maximum Pixel :", pixels.values.max())
    print("Mean Pixel    :", round(pixels.values.mean(), 2))
    print("Std Deviation :", round(pixels.values.std(), 2))