# Modular Two-Stage Network Intrusion Detection System for IoT

This repository is related to the paper "A Lean and Modular Two-Stage Network Intrusion Detection System for IoT Traffic" submitted to the IEEE Latincom 2024.


## Abstract
The popularization of the Internet of Things (IoT) has led to cyberattacks targeting interconnected applications. Traditional intrusion detection systems (IDS) struggle to cope with the increasing volume and complexity of IoT data, hindering their ability to identify all threats. In order to address this issue, we propose a modular two-stage Network IDS for IoT, with each stage specialized in a specific set of attacks. This approach allows for independent training and optimization of each stage, improving processing time and classification metrics when compared to single-stage systems. 
The effectiveness of this design is demonstrated using the CICIoT2023 dataset. Compared to a single model, our proposal obtains better inference time (13 seconds vs. 94 seconds) and an overall enhanced detection rate (0.9055 vs. 0.8370 in terms of F1-Score). An additional contribution of our work is the sharing of all developed code as open-source software, facilitating the reproduction and extension of our proposal.


## Requirements
- Python 3.12.3
- Install the required dependencies:  
    ```pip install -r requirements.txt```


## Project Structure
The project is organized into the following steps, each represented by a dedicated Jupyter notebook in the notebooks directory:

- **ETL** - `Notebook: 0 - ETL.ipynb`  
Purpose: Extract, Transform, and Load (ETL) process to gather all data from the source, consolidate it, and fix any data type issues or label inconsistencies.

- **EDA**: `Notebook: 1 - EDA.ipynb`  
Purpose: Perform Exploratory Data Analysis (EDA) to explore the dataset and analyze the relationships between features and the target variable.

- **Feature Engineering**: `Notebook: 2 - Feature Engineering.ipynb`  
Purpose: Preprocess the dataset with the aim of improving model performance. This involves creating, selecting, and transforming features.

- **Model**: `Notebook: 3 - Models.ipynb`  
Purpose: Develop and test a modular approach to Intrusion Detection. This step focuses on model design and evaluation using sample data.

- **Results**: `Notebook: 4 - Results.ipynb`  
Purpose: Measure and compare the model's performance against a baseline model and other published results, considering inference time, model size and classification metrics.


## Dataset
The project uses the [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset. Please refer to the paper for detailed instructions on dataset preparation.
