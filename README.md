# Automotive Diagnostic Action Recommender

# **Repository Overview:**
This repository features Python modules and Jupyter notebooks crafted during the 'Optimizing Repair Steps in Automotive Diagnostics with Deep Learning' project. The scripts provided cover various aspects of data handling, model design, training, testing, and performance evaluation.

## **File Descriptions:**

1. **data_preprocessing.py**: 
   - Houses all essential functions for data preprocessing tailored for training, testing, or prediction scenarios.

2. **process_data_parallel.py**: 
   - Incorporates the mechanism to divide expansive diagnostic data (up to 80 million rows) into manageable portions of 2 million rows each. These segments are saved in the 'chunks2m' directory.
   - Features functions for concurrent data preprocessing of the previously segmented chunks, making full use of the available CPU cores.

3. **cvf_da_model.py**: 
   - An operational script that formulates the 'Claims & Vehicle Fault-based Diagnostic Action Prediction (CVF-DA)' Model.
   - Facilitates training and validation using the preprocessed data.
   - Assesses the model's efficacy, presenting pertinent evaluation metrics and training trajectory.
  
4. **recommend_with_cvf_da_model.py**:
   - Utilises the trained CVF-DA model to produce recommendations by loading the model into memory.
  
6. **das_model.py**: 
   - An operational script that formulates the 'Diagnostic Action Sequence (DAS)' Model.
   - Facilitates training and validation using the preprocessed data.
   - Assesses the model's efficacy, presenting pertinent evaluation metrics and training trajectory.
