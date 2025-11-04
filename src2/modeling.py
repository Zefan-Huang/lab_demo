import pandas as pd
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    df_ms = pd.read_csv('unsupervised/cleaned_MS_data.csv')
    df_h = pd.read_csv('unsupervised/cleaned_data.csv')
    print("load them all")
    return df_ms, df_h

def combine_data(df_ms, df_h):
    df = pd.concat([df_ms, df_h],axis=0, ignore_index=True)
    df.to_csv('unsupervised/MS&health_data.csv', index=False)
    print("combined data")
    return df

def extract_features(df):
    features = []
    for id, sub in df.groupby('id'):
        x = sub['steps'].values
        mean_val = np.mean(x)
        std_val = np.std(x)
        # Safely compute ACF at lag up to 7; handle short series
        if len(x) > 1:
            nlags = min(7, len(x) - 1)
            acf_vals = acf(x, nlags=nlags, fft=False)
            acf_val = acf_vals[nlags]
        else:
            acf_val = np.nan
        low_ratio_val = np.mean(x < 0.3 * mean_val)
        high_ratio_val = np.mean(x > 1.7 * mean_val)

        days = np.arange(len(x))
        slope = linregress(days, x).slope if len(x) > 1 else 0.0
        trend_val = slope

        # Use a scalar for label (first value within the group)
        label_val = sub['label'].iloc[0] if 'label' in sub.columns and len(sub['label']) > 0 else np.nan

        features.append({'id': id,
                         'mean': mean_val,
                         'std': std_val,
                         'trend': trend_val,
                         'acf': acf_val,
                         'low_ratio': low_ratio_val,
                         'high_ratio': high_ratio_val,
                         'label': label_val})

    summary = pd.DataFrame(features)
    print(f'There are {len(summary)} IDs')
    print(summary.shape)
    print(summary.columns)
    return summary

def unsupervised_analysis(summary, n_cluaster, random_state):
    features = ['mean', 'std', 'trend', 'acf', 'low_ratio', 'high_ratio']
    X = summary[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Specify n_init to avoid warnings across sklearn versions
    kmeans = KMeans(n_clusters=n_cluaster, random_state=random_state, n_init='auto')
    summary['cluster'] = kmeans.fit_predict(X_scaled)

    print("\n cluster center")
    centers = pd.DataFrame(kmeans.cluster_centers_, columns = features)
    print(centers)

    print("\n each cluster number")
    print(summary['cluster'].value_counts().sort_index())

    ## MS vs Health
    crosstab = pd.crosstab(summary['cluster'].astype(str), summary['label'].astype(str), normalize='index')
    print(f"the ratio of clusters is: {crosstab}")

    #PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    summary['pca1'] = pca_result[:, 0]
    summary['pca2'] = pca_result[:, 1]

    score = silhouette_score(X_scaled, summary['cluster'])
    print(f"\n Silhouette Score: {score:.3f}")
    return summary

def plot(summary):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='pca1', y='pca2', hue='cluster', style='label',
        data=summary, palette='tab10', s=80, alpha=0.8
    )
    plt.title('KMeans Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(title='Cluster')
    plt.tight_layout()

    plt.savefig('unsupervised/cluster_lpot.png', dpi=500, bbox_inches='tight')
    print("plot saved")

def save_summary(summary, ):
    summary.to_csv('unsupervised/feed.csv', index=False)
    print('summary saved')

def main():
    df_ms, df_h = load_data()
    df = combine_data(df_ms, df_h)
    summary = extract_features(df)
    summary = unsupervised_analysis(summary, 3, 42)
    plot(summary)
    save_summary(summary)

if __name__ == '__main__':
    main()