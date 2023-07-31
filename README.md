### Postdoc at Scripps

All codes developed during postdoc at Scripps, around the in-situ data XXX

Environment:
XXX Add description of conda environment XXX

conda create -n mpl_niw -c conda-forge dask-jobqueue xarray zarr netcdf4 python-graphviz \
            fastparquet pyarrow bottleneck \
            jupyterlab jupyterhub ipywidgets notebook\
            xhistogram \
            cartopy geopandas \
            scikit-learn seaborn hvplot geoviews \
            intake-xarray gcsfs cmocean gsw \
            pytide pyinterp h3-py parcels

conda activate mpl_niw

pip install rechunker

conda install -c conda-forge xgcm xmitgcm

pip install -e .
