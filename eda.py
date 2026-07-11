import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Cấu hình đầu ra 
OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style chung 
PALETTE_MAIN  = "mako"
PALETTE_DIV   = "RdYlGn"
ACCENT        = "#4C72B0"
ACCENT2       = "#DD8452"
BG_COLOR      = "#F8F9FA"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor"  : BG_COLOR,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "font.family"     : "DejaVu Sans",
    "axes.titlesize"  : 14,
    "axes.labelsize"  : 12,
})

#  Đọc dữ liệu 
df = pd.read_csv("train.csv")
print(f"✔  Đọc dữ liệu: {df.shape[0]} dòng × {df.shape[1]} cột")

numeric_cols      = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols  = df.select_dtypes(include=["object"]).columns.tolist()
numeric_no_target = [c for c in numeric_cols if c not in ("Id", "SalePrice")]



# 1. TỔNG QUAN DỮ LIỆU – Missing values heatmap
def plot_missing_values():
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    pct  = (miss / len(df) * 100).round(2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Phân Tích Giá Trị Thiếu (Missing Values)", fontsize=16, fontweight="bold", y=1.01)

    #  Heatmap 
    miss_matrix = df[miss.index].isnull().astype(int)
    sns.heatmap(miss_matrix.T, cbar=False, yticklabels=True,
                cmap="Blues", ax=axes[0])
    axes[0].set_title("Heatmap vị trí giá trị thiếu")
    axes[0].set_xlabel("Chỉ số mẫu")

    #  Bar chart 
    colors = ["#E63946" if p > 40 else "#E9C46A" if p > 15 else ACCENT for p in pct]
    bars = axes[1].barh(miss.index, pct, color=colors)
    axes[1].set_xlabel("% thiếu")
    axes[1].set_title("Phần trăm thiếu theo cột")
    axes[1].invert_yaxis()
    for bar, val in zip(bars, pct):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_missing_values.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 2. PHÂN PHỐI BIẾN MỤC TIÊU – SalePrice
def plot_target_distribution():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phân Phối Biến Mục Tiêu – SalePrice", fontsize=16, fontweight="bold")

    #  (a) Histogram gốc
    ax = axes[0, 0]
    sns.histplot(df["SalePrice"], kde=True, color=ACCENT, ax=ax, bins=40)
    ax.set_title("(a) Phân phối SalePrice (gốc)")
    ax.set_xlabel("SalePrice ($)")
    skew_val = df["SalePrice"].skew()
    ax.axvline(df["SalePrice"].mean(), color="red", linestyle="--", label=f"Mean")
    ax.axvline(df["SalePrice"].median(), color="green", linestyle="--", label=f"Median")
    ax.legend()
    ax.text(0.95, 0.85, f"Skewness: {skew_val:.2f}", transform=ax.transAxes,
            ha="right", fontsize=11, color="darkred")

    #  (b) Log transform
    ax = axes[0, 1]
    log_price = np.log1p(df["SalePrice"])
    sns.histplot(log_price, kde=True, color=ACCENT2, ax=ax, bins=40)
    ax.set_title("(b) Phân phối log1p(SalePrice)")
    ax.set_xlabel("log1p(SalePrice)")
    ax.text(0.95, 0.85, f"Skewness: {log_price.skew():.2f}", transform=ax.transAxes,
            ha="right", fontsize=11, color="darkred")

    #  (c) QQ-plot gốc
    ax = axes[1, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(df["SalePrice"])
    ax.scatter(osm, osr, alpha=0.4, color=ACCENT, s=15)
    ax.plot([min(osm), max(osm)],
            [slope * min(osm) + intercept, slope * max(osm) + intercept],
            color="red", lw=2)
    ax.set_title("(c) QQ-Plot SalePrice (gốc)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    #  (d) QQ-plot log
    ax = axes[1, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(log_price)
    ax.scatter(osm, osr, alpha=0.4, color=ACCENT2, s=15)
    ax.plot([min(osm), max(osm)],
            [slope * min(osm) + intercept, slope * max(osm) + intercept],
            color="red", lw=2)
    ax.set_title("(d) QQ-Plot log1p(SalePrice)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_target_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 3. MA TRẬN TƯƠNG QUAN – Top 20 features
def plot_correlation_matrix():
    corr_with_target = df[numeric_cols].corr()["SalePrice"].abs().sort_values(ascending=False)
    top_cols = corr_with_target.head(21).index.tolist()
    corr = df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Ma Trận Tương Quan – Top 20 Đặc Trưng Số", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_correlation_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 4. TOP FEATURES vs SalePrice – Scatter plots
def plot_top_features_scatter():
    corr = df[numeric_no_target + ["SalePrice"]].corr()["SalePrice"].abs()
    top8 = corr.drop("SalePrice").sort_values(ascending=False).head(8).index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Tương Quan Các Đặc Trưng Quan Trọng vs SalePrice", fontsize=16, fontweight="bold")

    for ax, col in zip(axes.flatten(), top8):
        ax.scatter(df[col], df["SalePrice"], alpha=0.3, s=15, color=ACCENT)
        m, b, r, p, _ = stats.linregress(df[col].fillna(df[col].median()), df["SalePrice"])
        x_line = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(x_line, m * x_line + b, color="red", lw=2)
        ax.set_title(f"{col}\n(r = {r:.2f})", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("SalePrice")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_scatter_top_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 5. PHÂN PHỐI CÁC BIẾN SỐ QUAN TRỌNG
def plot_numeric_distributions():
    key_numerics = ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageArea",
                    "LotArea", "MasVnrArea", "OpenPorchSF", "WoodDeckSF"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Phân Phối Các Biến Số Quan Trọng", fontsize=16, fontweight="bold")

    for ax, col in zip(axes.flatten(), key_numerics):
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, color=ACCENT, bins=30)
        ax.set_title(f"{col}\nSkew={data.skew():.2f}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_numeric_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 6. CHẤT LƯỢNG & ĐIỀU KIỆN vs SalePrice
def plot_quality_vs_price():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Chất Lượng Tổng Thể & Điều Kiện vs SalePrice", fontsize=16, fontweight="bold")

    # OverallQual
    ax = axes[0]
    order = sorted(df["OverallQual"].unique())
    sns.boxplot(x="OverallQual", y="SalePrice", data=df, order=order, palette="mako", ax=ax)
    ax.set_title("OverallQual vs SalePrice")
    ax.set_xlabel("Overall Quality (1–10)")
    ax.set_ylabel("SalePrice ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    # OverallCond
    ax = axes[1]
    order = sorted(df["OverallCond"].unique())
    sns.boxplot(x="OverallCond", y="SalePrice", data=df, order=order, palette="rocket_r", ax=ax)
    ax.set_title("OverallCond vs SalePrice")
    ax.set_xlabel("Overall Condition (1–10)")
    ax.set_ylabel("SalePrice ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_quality_vs_price.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 7. KHU VỰC (NEIGHBORHOOD) vs SalePrice
def plot_neighborhood():
    order = df.groupby("Neighborhood")["SalePrice"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(y="Neighborhood", x="SalePrice", data=df, order=order,
                palette="mako", ax=ax, orient="h")
    ax.set_title("Khu Vực (Neighborhood) vs SalePrice", fontsize=16, fontweight="bold")
    ax.set_xlabel("SalePrice ($)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "07_neighborhood_vs_price.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 8. YẾU TỐ THỜI GIAN – Năm xây dựng & năm bán
def plot_time_features():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Phân Tích Theo Thời Gian", fontsize=16, fontweight="bold")

    # (a) YearBuilt vs SalePrice
    ax = axes[0]
    ax.scatter(df["YearBuilt"], df["SalePrice"], alpha=0.3, s=12, color=ACCENT)
    ax.set_title("(a) Năm Xây Dựng vs SalePrice")
    ax.set_xlabel("YearBuilt")
    ax.set_ylabel("SalePrice ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    # (b) YrSold – số lượng giao dịch
    ax = axes[1]
    yr_count = df["YrSold"].value_counts().sort_index()
    ax.bar(yr_count.index, yr_count.values, color=ACCENT2, width=0.6)
    ax.set_title("(b) Số Giao Dịch Theo Năm Bán")
    ax.set_xlabel("YrSold")
    ax.set_ylabel("Số lượng")

    # (c) YrSold vs median SalePrice
    ax = axes[2]
    yr_med = df.groupby("YrSold")["SalePrice"].median()
    ax.plot(yr_med.index, yr_med.values, marker="o", color=ACCENT, lw=2.5, markersize=8)
    ax.fill_between(yr_med.index, yr_med.values, alpha=0.15, color=ACCENT)
    ax.set_title("(c) Giá Trung Vị Theo Năm Bán")
    ax.set_xlabel("YrSold")
    ax.set_ylabel("Median SalePrice ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_time_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 9. BIẾN PHÂN LOẠI QUAN TRỌNG – Violin / Box
def plot_categorical_features():
    cat_features = ["MSZoning", "BldgType", "HouseStyle", "SaleCondition",
                    "KitchenQual", "GarageFinish"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Các Biến Phân Loại Quan Trọng vs SalePrice", fontsize=16, fontweight="bold")

    for ax, col in zip(axes.flatten(), cat_features):
        order = df.groupby(col)["SalePrice"].median().sort_values(ascending=False).index
        sns.violinplot(x=col, y="SalePrice", data=df, order=order,
                       palette="mako", ax=ax, inner="box", cut=0)
        ax.set_title(f"{col}")
        ax.set_xlabel("")
        ax.set_ylabel("SalePrice ($)")
        ax.tick_params(axis="x", rotation=30)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_categorical_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 10. OUTLIER ANALYSIS – GrLivArea & SalePrice
def plot_outlier_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phân Tích Outlier", fontsize=16, fontweight="bold")

    # (a) GrLivArea vs SalePrice – highlight outliers
    ax = axes[0]
    outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)]
    normal   = df.drop(outliers.index)
    ax.scatter(normal["GrLivArea"], normal["SalePrice"], alpha=0.3, s=15,
               color=ACCENT, label="Normal")
    ax.scatter(outliers["GrLivArea"], outliers["SalePrice"], alpha=0.9, s=60,
               color="red", zorder=5, label=f"Outlier (n={len(outliers)})")
    ax.axvline(4000, color="orange", linestyle="--", lw=1.5, label="GrLivArea=4000")
    ax.axhline(300000, color="purple", linestyle="--", lw=1.5, label="Price=300K")
    ax.set_title("(a) GrLivArea vs SalePrice (Outliers)")
    ax.set_xlabel("GrLivArea (sq ft)")
    ax.set_ylabel("SalePrice ($)")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    # (b) IQR Boxplot cho top numeric features
    ax = axes[1]
    key = ["GrLivArea", "TotalBsmtSF", "GarageArea", "LotArea", "1stFlrSF"]
    df_norm = df[key].apply(lambda c: (c - c.mean()) / c.std())
    df_norm.boxplot(ax=ax, patch_artist=True,
                    boxprops=dict(facecolor=ACCENT, alpha=0.6),
                    medianprops=dict(color="red", lw=2),
                    whiskerprops=dict(color="gray"),
                    capprops=dict(color="gray"),
                    flierprops=dict(marker="o", markersize=3, alpha=0.3, color="gray"))
    ax.set_title("(b) Z-Score Boxplot (top diện tích)")
    ax.set_ylabel("Z-Score")
    ax.tick_params(axis="x", rotation=15)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "10_outlier_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 11. FEATURE ENGINEERING – Đặc trưng mới
def plot_engineered_features():
    df2 = df.copy()
    df2["TotalSF"]   = df2["TotalBsmtSF"].fillna(0) + df2["1stFlrSF"].fillna(0) + df2["2ndFlrSF"].fillna(0)
    df2["TotalBath"] = (df2["FullBath"].fillna(0) +
                        0.5 * df2["HalfBath"].fillna(0) +
                        df2["BsmtFullBath"].fillna(0) +
                        0.5 * df2["BsmtHalfBath"].fillna(0))
    df2["HouseAge"]  = df2["YrSold"] - df2["YearBuilt"]
    df2["RemodAge"]  = df2["YrSold"] - df2["YearRemodAdd"]
    df2["TotalQual"] = df2["OverallQual"] + df2["OverallCond"]

    eng_features = ["TotalSF", "TotalBath", "HouseAge", "RemodAge", "TotalQual"]

    fig, axes = plt.subplots(1, len(eng_features), figsize=(20, 5))
    fig.suptitle("Đặc Trưng Mới (Feature Engineering) vs SalePrice", fontsize=16, fontweight="bold")

    colors = [ACCENT, ACCENT2, "#2A9D8F", "#E76F51", "#8338EC"]
    for ax, col, c in zip(axes, eng_features, colors):
        ax.scatter(df2[col], df2["SalePrice"], alpha=0.3, s=12, color=c)
        m, b, r, _, _ = stats.linregress(df2[col].fillna(0), df2["SalePrice"])
        x_line = np.linspace(df2[col].min(), df2[col].max(), 100)
        ax.plot(x_line, m * x_line + b, color="black", lw=2)
        ax.set_title(f"{col}\n(r={r:.2f})", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("SalePrice ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "11_engineered_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# 12. TỔNG QUAN THỐNG KÊ – Summary dashboard
def plot_summary_dashboard():
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Dashboard Tổng Quan", fontsize=18, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # (a) Số loại cột
    ax1 = fig.add_subplot(gs[0, 0])
    counts = [len(numeric_no_target), len(categorical_cols)]
    wedge_colors = [ACCENT, ACCENT2]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=["Numeric", "Categorical"],
        autopct="%1.0f%%", colors=wedge_colors,
        startangle=90, wedgeprops=dict(edgecolor="white", lw=2))
    ax1.set_title("(a) Phân loại đặc trưng")

    # (b) Missing rate distribution
    ax2 = fig.add_subplot(gs[0, 1])
    miss_pct = df.isnull().mean() * 100
    miss_pct = miss_pct[miss_pct > 0]
    bins_m = [0, 5, 20, 50, 100]
    labels_m = ["<5%", "5–20%", "20–50%", ">50%"]
    miss_cat = pd.cut(miss_pct, bins=bins_m, labels=labels_m)
    miss_cat.value_counts().reindex(labels_m).plot(kind="bar", ax=ax2,
                                                    color=["#2A9D8F", "#E9C46A", "#F4A261", "#E63946"])
    ax2.set_title("(b) Phân nhóm % missing")
    ax2.set_ylabel("Số cột")
    ax2.tick_params(axis="x", rotation=0)

    # (c) Skewness phân phối
    ax3 = fig.add_subplot(gs[0, 2:])
    skew_vals = df[numeric_no_target].skew().sort_values(ascending=False).head(15)
    colors_sk = ["#E63946" if s > 1 else "#F4A261" if s > 0.5 else ACCENT for s in skew_vals]
    skew_vals.plot(kind="bar", ax=ax3, color=colors_sk)
    ax3.set_title("(c) Skewness Top 15 đặc trưng số")
    ax3.set_ylabel("Skewness")
    ax3.axhline(1, color="red", lw=1, linestyle="--", label=">1 (highly skewed)")
    ax3.axhline(0.5, color="orange", lw=1, linestyle="--", label=">0.5")
    ax3.legend(fontsize=8)
    ax3.tick_params(axis="x", rotation=45)

    # (d) Correlation bar chart with SalePrice
    ax4 = fig.add_subplot(gs[1, :2])
    corr_sp = df[numeric_cols].corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False)
    top_pos = corr_sp.head(10)
    top_neg = corr_sp.tail(5)
    combined = pd.concat([top_pos, top_neg])
    colors_corr = [ACCENT if v > 0 else "#E63946" for v in combined]
    combined.plot(kind="bar", ax=ax4, color=colors_corr)
    ax4.set_title("(d) Tương quan với SalePrice")
    ax4.set_ylabel("Pearson r")
    ax4.axhline(0, color="black", lw=0.8)
    ax4.tick_params(axis="x", rotation=45)

    # (e) Phân phối SalePrice theo MoSold
    ax5 = fig.add_subplot(gs[1, 2:])
    month_med = df.groupby("MoSold")["SalePrice"].median()
    ax5.bar(month_med.index, month_med.values, color=ACCENT2, alpha=0.85)
    ax5.set_title("(e) Giá Trung Vị Theo Tháng Bán")
    ax5.set_xlabel("Tháng")
    ax5.set_ylabel("Median SalePrice")
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax5.set_xticks(range(1, 13))

    path = os.path.join(OUTPUT_DIR, "12_summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# MAIN
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BẮT ĐẦU PHÂN TÍCH DỮ LIỆU KHÁM PHÁ (EDA)")
    print("=" * 60)

    steps = [
        ("Giá trị thiếu",             plot_missing_values),
        ("Phân phối SalePrice",        plot_target_distribution),
        ("Ma trận tương quan",         plot_correlation_matrix),
        ("Scatter top features",       plot_top_features_scatter),
        ("Phân phối biến số",          plot_numeric_distributions),
        ("Chất lượng vs giá",         plot_quality_vs_price),
        ("Khu vực vs giá",            plot_neighborhood),
        ("Yếu tố thời gian",          plot_time_features),
        ("Biến phân loại",            plot_categorical_features),
        ("Phân tích Outlier",         plot_outlier_analysis),
        ("Đặc trưng mới",             plot_engineered_features),
        ("Dashboard tổng quan",        plot_summary_dashboard),
    ]

    for i, (name, fn) in enumerate(steps, 1):
        print(f"\n[{i:02d}/{len(steps)}] {name} ...")
        fn()

    print("\n" + "=" * 60)
    print(f"  ✔  HOÀN TẤT! {len(steps)} sơ đồ đã được lưu vào: ./{OUTPUT_DIR}/")
    print("=" * 60)
