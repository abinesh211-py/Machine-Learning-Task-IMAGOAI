# Prediction of Mycotoxin Levels in Corn using CNN on Hyperspectral Imaging Data

This project focuses on predicting mycotoxin (vomitoxin) levels in corn samples using hyperspectral imaging and deep learning techniques. A Convolutional Neural Network (CNN) was developed and optimized for effective hyperspectral data processing.

## Files in Repository
- `mlTask1.ipynb` – Jupyter Notebook for data processing, model training, and evaluation.
- `report.pdf` – Short report covering preprocessing, model architecture, results, and insights.
- `README.md` – Project documentation (this file).

## Installation
### Prerequisites
Ensure you have Python 3.8+ and install required libraries:
```bash
pip install -r requirements.txt
```
#### Required Libraries
The following Python libraries are used in this project:
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
```
Alternatively, install them manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

## Dataset and Preprocessing
- The dataset contains hyperspectral reflectance values across **448 spectral bands** for different corn samples.
- Target variable: **vomitoxin concentration (in ppb)**.
- Preprocessing steps:
  - **Standardization:** Applied using `StandardScaler`.
  - **Dimensionality Reduction:** Principal Component Analysis (PCA) reduced features from **448 to 30** components, capturing **98.7% variance**.
  - **Data Visualization:**
    - Line plots for average spectral reflectance.
    - Heatmaps for sample comparison.
  - **Data Splitting:** 80% training, 20% testing.
  - **Reshaping for CNN:** Data structured for 1D CNN input.

## Model Development
### CNN Architecture
- **Conv1D layers:** 128, 64, 32 filters.
- **Kernel sizes:** 9, 5, 3.
- **Regularization:** Batch Normalization and Dropout.
- **Optimizer:** Adam with a learning rate of **0.0003**.
- **Training:** 200 epochs, batch size of 8.

### Performance Metrics
Final model achieved:
- **Root Mean Squared Error (RMSE):** 3948.47
- **Coefficient of Determination (R²):** 0.9442

## Baseline Model Comparison
To assess the CNN model's effectiveness, a simple **Linear Regression** model was tested:
- **RMSE:** 6203.21
- **R² Score:** 0.8125

This comparison highlights CNN’s superior predictive performance.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/mycotoxin-prediction.git
cd mycotoxin-prediction
```
2. Open and run `mlTask1.ipynb` in Jupyter Notebook.
3. Review `report.pdf` for detailed findings.

## Key Findings
- PCA significantly reduced computational complexity while retaining most variance.
- Kernel size optimization and batch size reduction significantly improved performance.
- Learning rate fine-tuning and increased epochs enhanced model convergence.
- **Future Work:**
  - Exploring **hybrid models (CNN + LSTM)**.
  - Implementing **attention mechanisms** for feature importance analysis.
  - Creating a **Streamlit app** for interactive user predictions.

## Contact
For any queries, contact Abinesh at [abineshpcm@gmail.com].

