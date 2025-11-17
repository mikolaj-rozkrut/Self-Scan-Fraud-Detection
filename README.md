# Supermarket Self-Checkout Fraud Detection

This repository contains the data and code for a project to build and evaluate a machine learning model that flags potentially fraudulent transactions at supermarket self-checkout counters. The primary goal is to create a model that can analyze customer purchase data to identify behavior indicative of fraud, such as not scanning all items.

The main analysis, feature engineering, and modeling process is detailed in the `Fraud-Detection-Project.ipynb` Jupyter Notebook.

## 📖 Table of Contents
* [About The Project](#about-the-project)
* [Datasets](#datasets)
* [Project Notebook](#project-notebook-fraud-detection-projectipynb)
* [How to Run](#how-to-run)
* [Note on the Prediction Pipeline](#note-on-the-prediction-pipeline)

## 🎯 About The Project

This project treats the problem of fraud detection as a classification task. Based on a series of scanned items, the model aims to predict whether a transaction is fraudulent or legitimate.

The core of the project lies in feature engineering, where raw transaction data (like scan timestamps and product categories) is transformed into meaningful features that can capture suspicious behavior. These features are then used to train a model to identify patterns associated with fraud.

## 💾 Datasets

All data for this project is available in the `/data` folder.

* `Product_Catalog.csv`: Contains detailed information for all products available in the store, such as `id`, `category`, `sub category`, `product`, and `price`.
* `Purchases.csv`: Provides the raw transaction data. Each row represents a single scanned item, linking a `purchase id` to a `product id` and the `time` of the scan (in seconds).
* `Labels.csv`: Contains the ground truth for a subset of transactions. It maps a `purchase id` to a specific monetary `amount`, which indicates the value of unscanned (fraudulent) items. In the notebook, this is converted to a binary `Fraud` label (1 if `amount` > 0, 0 otherwise).

Additionally, the `/real_time_data` folder contains sample files that can be used to test the project's prediction pipeline.

## 💻 Project Notebook: Fraud-Detection-Project.ipynb

The main notebook details the entire process of building the fraud detection model. Key steps include:

1.  **Data Loading & Cleaning:** Importing the three core CSV files and checking for missing values.
2.  **Feature Engineering:** This is a crucial step where new, predictive features are created from the raw data. These include:
    * Total price and average price per transaction.
    * Total number of products and distinct categories per transaction.
    * Total scan time and average scan time per item.
    * **Scanning Abnormality:** A key feature that models the average time between scanning items based on their product categories. Scans that take significantly longer than the average are flagged as potential abnormalities.
3.  **Model Training:** The engineered features are used to train various classification models, including an LSTM-based neural network, to predict the `Fraud` label.
4.  **Prediction Pipeline:** The end of the notebook features a complete pipeline that can take new data (like the files in `/real_time_data`), process it, and generate a fraud prediction.

## 🚀 How to Run

This project is set up to run in Google Colab.

1.  **Clone the repository or download the files.**
2.  **Upload to Google Drive:**
    * Upload the `Fraud-Detection-Project.ipynb` notebook to your Google Drive.
    * Upload the `data` folder (containing `Product_Catalog.csv`, `Purchases.csv`, `Labels.csv`) and the `real_time_data` folder to your Drive.
3.  **Open in Colab:** Open the `Fraud-Detection-Project.ipynb` notebook with Google Colaboratory.
4.  **Update File Paths:** In the first code cell ("1. READING THE DATA"), update the file paths for `purchases_file`, `product_catalog_file`, and `labels_file` to match their locations in your Google Drive.
5.  **Run Notebook:** You can now run the cells sequentially to execute the full analysis, from data loading and feature engineering to model training and evaluation.

### Dependencies

The project relies on standard Python data science libraries:
* `pandas`
* `numpy`
* `seaborn`
* `matplotlib`
* `tensorflow` (for the LSTM model)
* `scikit-learn`

## ⚠️ Note on the Prediction Pipeline

Please note that this project was originally designed to send its predictions to an external tester page for validation. **Access to this external tester page is no longer available.**

However, the complete code for the prediction pipeline is intentionally left at the end of the notebook. This code serves as a demonstration of a complete, end-to-end workflow—showing how new data can be loaded, processed, and scored by the trained model. You can use the files in the `/real_time_data` folder to test this pipeline's functionality.
