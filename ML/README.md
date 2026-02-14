conda install -c conda-forge geopandas rasterio shapely pyproj fiona


make_grid:-  python src/make_grid.py --raster data/raw/sentinel/S2_2020_Jun01_Aug31_10m_QA60SCL_F32.tif --cell-size 100 --block-size 1000 --output data/processed/grid/grid_100m.gpkg

extract_feature:-
1->python src/extract_features.py --raster data/raw/sentinel/S2_2019_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2019.parquet --year 2019
2-> python src/extract_features.py --raster data/raw/sentinel/S2_2020_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2020.parquet --year 2020
3-> python src/extract_features.py --raster data/raw/sentinel/S2_2021_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2021.parquet --year 2021
4-> python src/extract_features.py --raster data/raw/sentinel/S2_2022_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2022.parquet --year 2022
5-> python src/extract_features.py --raster data/raw/sentinel/S2_2023_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2023.parquet --year 2023


extract_labels:- 
1-> python src/extract_labels.py --raster data/raw/worldcover/WorldCover_2020_10m.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/labels/labels_2020.parquet --year 2020
2-> python src/extract_labels.py --raster data/raw/worldcover/WorldCover_2021_10m.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/labels/labels_2021.parquet --year 2021

build_training_table:- 
python src/build_training_table.py --grid data/processed/grid/grid_100m.gpkg --features-dir data/processed/features --labels-dir data/processed/labels --years 2020 2021 --output data/processed/tables/train_table.parquet

train_models:- 
python src/train_models.py --train data/processed/tables/train_table.parquet --outdir models --spatial-folds 5 --optuna-trials 30 --ensemble-size 5

predict_all_years:- 
python src/predict_all_years.py --features-dir data/processed/features --years 2019 2020 2021 2022 2023 --model-dir models --output-dir data/processed/predictions --include-uncertainty



