import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to import tslearn; if unavailable or incompatible with sklearn, fall back to numpy/sklearn
try:
    from tslearn.clustering import TimeSeriesKMeans  # type: ignore
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance  # type: ignore
    _TSLEARN_AVAILABLE = True
except Exception:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    _TSLEARN_AVAILABLE = False

    class TimeSeriesScalerMeanVariance:
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X = np.asarray(X, dtype=float)
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            return (X - mean) / std
        def fit_transform(self, X, y=None):
            return self.transform(X)


def load_data():
    dfo = pd.read_csv('unsupervised/MS&health_data.csv')
    return dfo


def data_preprocessing(dfo, target_len):
    dfo['id'] = dfo['id'].astype(str)
    df = dfo[dfo['id'] != '3']
    print('ID = 3 has been deleted, because the number of days are too short.')

    all_series = []
    for pid, sub in df.groupby('id'):
        x = sub['steps'].astype(float).values
        x = x[:target_len]
        label = sub['label'].iloc[0] if 'label' in sub.columns and len(sub['label']) > 0 else np.nan
        all_series.append({'id': pid, 'x': x, 'label': label})

    df_out = pd.DataFrame(all_series)
    print(f"There are {len(df_out)} ids and each length is {target_len}.")
    return df_out


def dtw_cluster(df_out, n_clusters, random_state):
    x = np.stack(df_out['x'].values)
    x = x[..., np.newaxis]

    scaler = TimeSeriesScalerMeanVariance()
    x_scaled = scaler.fit_transform(x)

    if _TSLEARN_AVAILABLE:
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', random_state=random_state)
        labels = model.fit_predict(x_scaled)
    else:
        # Fallback: use plain KMeans on flattened standardized series (Euclidean)
        print('Warning: tslearn is unavailable or incompatible with your scikit-learn version. '
              'Falling back to Euclidean KMeans on standardized series.')
        X_flat = x_scaled.reshape(x_scaled.shape[0], -1)
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        labels = model.fit_predict(X_flat)

    df_out = df_out.copy()
    df_out['cluster'] = labels
    print('Finished clustering')
    print(df_out['cluster'].value_counts())

    if 'label' in df_out.columns:
        print('MS vs Healthy ratio')
        print(pd.crosstab(df_out['cluster'], df_out['label'], normalize='index'))

    return df_out, model


def plot_clusters(df_out, title="DTW Clustering"):
    plt.figure(figsize=(10, 6))
    for cluster_id in sorted(df_out['cluster'].unique()):
        cluster_series = np.stack(df_out[df_out['cluster'] == cluster_id]['x'])
        mean_curve = cluster_series.mean(axis=0)
        plt.plot(mean_curve, label=f'Cluster {cluster_id}', linewidth=2)

    for cluster_id in sorted(df_out['cluster'].unique()):
        for i, row in df_out[df_out['cluster'] == cluster_id].iterrows():
            if row['label'] == 0:
                plt.plot(row['x'], color='blue', alpha=0.15, linestyle='--')  # MS (label=0)
            else:
                plt.plot(row['x'], color='red', alpha=0.1, linestyle=':')  # Healthy (label=1)

    plt.title(title)
    plt.xlabel(f'Day (0–{len(df_out["x"].iloc[0]) - 1})')
    plt.ylabel('Steps (standardized)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('unsupervised/DTW.png', dpi=500, bbox_inches='tight')
    plt.show()


def save(df_out):
    df_out.to_csv('unsupervised/cleaned_data.csv', index=False)
    print('Data saved')


def main():
    dfo = load_data()
    df_out = data_preprocessing(dfo, target_len=66)
    df_out, model = dtw_cluster(df_out, n_clusters=3, random_state=42)
    plot_clusters(df_out, title="Time Series Clustering")
    save(df_out)


if __name__ == '__main__':
    main()