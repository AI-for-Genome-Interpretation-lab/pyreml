import pandas as pd
import os

_pkg_dir = os.path.dirname(__file__)
_data_dir = os.path.join(_pkg_dir, "data")

larix = pd.read_csv(os.path.join(_data_dir, "Larix.csv"))