import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
import os
import zipfile
import shutil

# ==========================================
# PART 0: DOWNLOAD DATASET FROM KAGGLE
# ==========================================
script_dir = Path(__file__).parent
DATA_PATH = script_dir / "DatasetSaham"
DAILY_PATH = DATA_PATH / "daily"

def download_kaggle_dataset():
    """
    Download IHSG Stock Data from Kaggle
    Dataset: https://www.kaggle.com/datasets/muamkh/ihsgstockdata
    """
    # Check if dataset already exists
    if DAILY_PATH.exists() and len(list(DAILY_PATH.glob("*.csv"))) > 0:
        print("Dataset already exists. Skipping download.")
        return True
    
    
    # Try using opendatasets (easier, no API key needed for public datasets)
    try:
        import opendatasets as od
        
        # Download dataset
        print("Downloading using opendatasets...")
        od.download("https://www.kaggle.com/datasets/muamkh/ihsgstockdata", 
                    data_dir=str(script_dir))
        
        # Move files to correct location
        downloaded_path = script_dir / "ihsgstockdata"
        if downloaded_path.exists():
            # Create DatasetSaham folder if not exists
            DATA_PATH.mkdir(exist_ok=True)
            
            # Move contents
            for item in downloaded_path.iterdir():
                dest = DATA_PATH / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(DATA_PATH))
            
            # Remove empty downloaded folder
            if downloaded_path.exists():
                shutil.rmtree(downloaded_path)
            
            print("\nDataset downloaded and extracted successfully!")
            return True
            
    except ImportError:
        print("opendatasets not installed. Trying kaggle API...")
    except Exception as e:
        print(f"opendatasets failed: {e}")
        print("Trying kaggle API...")
    
    # Try using kaggle API
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("Downloading using Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        api.dataset_download_files('muamkh/ihsgstockdata', 
                                   path=str(script_dir), 
                                   unzip=True)
        
        # Move files to correct location
        downloaded_path = script_dir / "ihsgstockdata"
        if downloaded_path.exists():
            DATA_PATH.mkdir(exist_ok=True)
            for item in downloaded_path.iterdir():
                dest = DATA_PATH / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(DATA_PATH))
            shutil.rmtree(downloaded_path)
        
        print("\nDataset downloaded and extracted successfully!")
        return True
        
    except ImportError:
        print("\nERROR: Neither 'opendatasets' nor 'kaggle' package is installed.")
        return False
        
    except Exception as e:
        print(f"\nERROR downloading dataset: {e}")
        print("\nManual download instructions:")
        print("  1. Go to: https://www.kaggle.com/datasets/muamkh/ihsgstockdata")
        print("  2. Click 'Download' button")
        print("  3. Extract the zip file")
        print(f"  4. Place the contents in: {DATA_PATH}")
        return False

# Download dataset if needed
if not download_kaggle_dataset():
    print("\nPlease download the dataset manually and run again.")
    exit(1)

# ==========================================
# PART 1: LOAD REAL STOCK DATA (DAILY DATASET)
# ==========================================
print("\n" + "="*60)
print("LOADING STOCK DATA")
print("="*60)

# Load stock list to get sector information
df_daftar = pd.read_csv(DATA_PATH / "DaftarSaham.csv")
print("Available Sectors:")
print(df_daftar['Sector'].value_counts())

# Select 4 main sectors for clustering: Financials, Energy, Infrastructures, Technology
SELECTED_SECTORS = ['Financials', 'Energy', 'Infrastructures', 'Technology']
SECTOR_LABELS = ['Financials', 'Energy', 'Infrastructures', 'Technology']

# Filter stocks based on selected sectors
df_filtered = df_daftar[df_daftar['Sector'].isin(SELECTED_SECTORS)].copy()

# Sample stocks from each sector (max 70 per sector for efficiency)
stocks_per_sector = 70
selected_stocks = []
sector_labels = []
stock_sectors = []

for sector in SELECTED_SECTORS:
    sector_stocks = df_filtered[df_filtered['Sector'] == sector]['Code'].tolist()[:stocks_per_sector]
    selected_stocks.extend(sector_stocks)
    sector_labels.extend([SELECTED_SECTORS.index(sector)] * len(sector_stocks))
    stock_sectors.extend([sector] * len(sector_stocks))

print(f"\nTotal stocks selected: {len(selected_stocks)}")
print(f"Distribution per sector: {dict(zip(SELECTED_SECTORS, [sector_labels.count(i) for i in range(len(SELECTED_SECTORS))]))}")

# ==========================================
# PART 2: READ OHLC DATA FROM CSV
# ==========================================
n_days = 252  # 1 trading year (252 business days)
n_features = 4  # Open, High, Low, Close
stock_data = {}
valid_stocks = []
valid_sectors = []
valid_sector_labels = []

for i, stock_code in enumerate(selected_stocks):
    csv_path = DAILY_PATH / f"{stock_code}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Get last n_days
            if len(df) >= n_days:
                df_last = df.tail(n_days).copy()
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close']
                if all(col in df.columns for col in required_cols):
                    stock_data[stock_code] = df_last[required_cols].values
                    valid_stocks.append(stock_code)
                    # Store sector label
                    sector_idx = SELECTED_SECTORS.index(stock_sectors[i])
                    valid_sectors.append(SECTOR_LABELS[sector_idx])
                    valid_sector_labels.append(sector_labels[i])
        except Exception as e:
            print(f"Error reading {stock_code}: {e}")

n_stocks = len(valid_stocks)
print(f"\nNumber of valid stocks successfully loaded: {n_stocks}")

# Create 3D TENSOR (Stocks x Time x Features)
tensor_data = np.zeros((n_stocks, n_days, n_features))
for i, stock_code in enumerate(valid_stocks):
    tensor_data[i] = stock_data[stock_code]

print(f"Tensor Shape: {tensor_data.shape}")

# ==========================================
# PART 3: PREPROCESSING (NORMALIZATION)
# ==========================================
# Important: Standardize stock price scales (Z-Score Normalization)
tensor_norm = np.zeros_like(tensor_data)
for i in range(n_stocks):
    for f in range(n_features):
        series = tensor_data[i, :, f]
        std = np.std(series)
        if std > 0:
            tensor_norm[i, :, f] = (series - np.mean(series)) / std
        else:
            tensor_norm[i, :, f] = 0

# ==========================================
# PART 4: HOSVD ALGORITHM (CORE MATHEMATICS)
# ==========================================
# Step 1: Mode-1 Unfolding (Flatten Tensor to Matrix)
matrix_unfolded = np.reshape(tensor_norm, (n_stocks, n_days * n_features))
print(f"Matrix size after Unfolding: {matrix_unfolded.shape}")

# Step 2: Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(matrix_unfolded, full_matrices=False)

# Step 3: Dimensionality Reduction
k_factors = min(len(SELECTED_SECTORS), n_stocks - 1, 10)
stock_features = U[:, :k_factors]

print(f"Latent Stock Features Dimension: {stock_features.shape}")

# ==========================================
# PART 5: K-MEANS CLUSTERING
# ==========================================
n_clusters = len(SELECTED_SECTORS)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(stock_features)

# ==========================================
# PART 6: VISUALIZATION OF RESULTS
# ==========================================
fig = plt.figure(figsize=(16, 10))

# Plot 1: 2D Stock Distribution (PC1 vs PC2)
ax1 = fig.add_subplot(2, 2, 1)
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
markers = ['o', 's', '^', 'D', 'v', 'p']

for i in range(n_stocks):
    ax1.scatter(stock_features[i, 0], stock_features[i, 1],
                c=colors[labels[i] % len(colors)],
                marker=markers[valid_sector_labels[i] % len(markers)],
                s=100, alpha=0.7, edgecolor='k')
    ax1.text(stock_features[i, 0]+0.002, stock_features[i, 1]+0.002,
             valid_stocks[i], fontsize=6)

ax1.set_title('Stock Clustering with HOSVD \nColor=Predicted Cluster, Shape=Original Sector')
ax1.set_xlabel('Principal Component 1 (Latent Factor 1)')
ax1.set_ylabel('Principal Component 2 (Latent Factor 2)')
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Confusion Matrix (Heatmap)
ax3 = fig.add_subplot(2, 2, 3)
confusion = np.zeros((n_clusters, len(SELECTED_SECTORS)))
for pred, true in zip(labels, valid_sector_labels):
    confusion[pred, true] += 1

im = ax3.imshow(confusion, cmap='Blues')
ax3.set_xticks(np.arange(len(SELECTED_SECTORS)))
ax3.set_yticks(np.arange(n_clusters))
ax3.set_xticklabels(SECTOR_LABELS, rotation=45, ha='right')
ax3.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)])
ax3.set_xlabel('Original Sector')
ax3.set_ylabel('Predicted Cluster')
ax3.set_title('Confusion Matrix')

# Add values in each cell
for i in range(n_clusters):
    for j in range(len(SELECTED_SECTORS)):
        text = ax3.text(j, i, int(confusion[i, j]), ha="center", va="center",
                       color="white" if confusion[i, j] > confusion.max()/2 else "black")

plt.colorbar(im, ax=ax3)

plt.tight_layout()
plt.savefig('clustering_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# PART 7: DISPLAY DETAILED RESULTS
# ==========================================
df_result = pd.DataFrame({
    'Stock_Code': valid_stocks,
    'Original_Sector': valid_sectors,
    'Predicted_Cluster': labels,
    'PC1': stock_features[:, 0],
    'PC2': stock_features[:, 1],
})

print("\n" + "="*60)
print("STOCK CLASSIFICATION RESULTS USING HOSVD + K-MEANS")
print("="*60)
print(f"\nEvaluation Metrics:")
print(f"  - Number of Stocks: {n_stocks}")
print(f"  - Number of Clusters: {n_clusters}")
print(f"  - Data Duration: {n_days} trading days")

print("\n" + "-"*60)
print("STOCK LIST PER CLUSTER:")
print("-"*60)

for cluster_id in range(n_clusters):
    cluster_stocks = df_result[df_result['Predicted_Cluster'] == cluster_id]
    print(f"\n[CLUSTER {cluster_id}] ({len(cluster_stocks)} stocks)")
    
    # Calculate dominant sector in this cluster
    sector_dist = cluster_stocks['Original_Sector'].value_counts()
    dominant_sector = sector_dist.index[0] if len(sector_dist) > 0 else "N/A"
    print(f"  Dominant Sector: {dominant_sector}")
    print(f"  Distribution: {dict(sector_dist)}")
    print(f"  Stocks: {', '.join(cluster_stocks['Stock_Code'].tolist())}")

# Save to CSV
print("Chart saved to: clustering_results.png")

