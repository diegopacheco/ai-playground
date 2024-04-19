import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Download the FER2013 dataset
api.dataset_download_files('msambare/fer2013', path='./', unzip=True)

# The dataset is downloaded as a zip file, so we need to extract it
with zipfile.ZipFile('./fer2013.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

# Now the dataset is available at './fer2013.csv'
data = pd.read_csv('./fer2013.csv')