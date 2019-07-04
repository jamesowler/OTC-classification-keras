# Optical Coherence Tomography Classifier

James Owler

---

Dataset: https://www.kaggle.com/paultimothymooney/kermany2018


---

## To train classifier:

#### Clone git repository
`git clone https://github.com/jamesowler/OTC-classification-keras.git .`

`cd OTC-classification-keras`


#### Install requirements

`pip install -r requirements`


#### Create directory for the dataset
`mkdir data`


#### Download dataset

Option 1: manually download the data using a web browser

Option 2: Download data using kaggle API:
 
`pip install kaggle`

Create new API token from your in your kaggle account settings 

`kaggle datasets download -d paultimothymooney/kermany2018 -p ./data`


#### unzip data
`unzip kermany2018.zip`
 
`unzip OCT2017.zip`


#### Run training
`python train.py`

---


