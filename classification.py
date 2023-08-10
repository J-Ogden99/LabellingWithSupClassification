from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

pd.set_option('display.max_columns', 20)

original_df = pd.read_parquet('for_classification_aug10_nogeom.parquet')
df = original_df.loc[~(original_df['x'].isna())]

# Run region-by-region, so streams from outside of a region
# don't get clustered with a VPU Code that belongs to that region
for region in df['TDXHydroRegion'].unique():
    df_temp = df.loc[df['TDXHydroRegion'] == region]
    df_pred = df_temp.loc[df_temp.VPUCode.isna()]
    df_temp = df_temp.loc[~df_temp.VPUCode.isna()]

    if df_pred.empty or df_temp.empty:
        continue
    X = df_temp[['x', 'y']].values
    y = df_temp['VPUCode'].values
    X_pred = df_pred[['x', 'y']].values

    # del df_temp

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, y)

    y_pred = neigh.predict(X_pred)

    df_pred['VPUCode_new'] = y_pred

    original_df = original_df.merge(df_pred[['TDXHydroLinkNo', 'VPUCode_new']], on='TDXHydroLinkNo', how='left')
    original_df['VPUCode'] = original_df['VPUCode'].fillna(original_df['VPUCode_new'])
    original_df.drop('VPUCode_new', axis=1, inplace=True)

original_df.to_parquet('classified_rivers.parquet')
# exit()
point_geometries = [Point(x, y) for x, y in zip(original_df['x'], original_df['y'])]

original_df = gpd.GeoDataFrame(original_df, geometry=point_geometries)

original_df.to_file('classified_rivers.gpkg', driver='GPKG')
