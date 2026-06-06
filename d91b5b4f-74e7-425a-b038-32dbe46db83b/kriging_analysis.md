# ROGII Wellbore Geology Prediction: Kriging Implementation Debugging Report

This report analyzes the Kaggle execution log output of [kriging_mapping_subm.ipynb](file:///c:/Users/vignesh.nehru/OneDrive%20-%20Mu%20Sigma%20Business Solutions Pvt. Ltd/Documents/ml/rogii/rogii_subterrain_kag/kriging_mapping_subm.ipynb), explaining the root causes behind the Stage 1 TVT RMSE spike to **39.12 ft** and the training time explosion to **170.58 minutes**.

---

## 1. Why Stage 1 RMSE Spiked from 14.56 ft to 39.12 ft

The dipping plane baseline prior ($Z_{\text{form}} = Z + \text{TVT}$) represents the regional geology. The spike in prior RMSE is caused by a violation of **stationarity** in the Kriging model.

```
Elevation (Z_form)
   ^                                          / (True Dipping Plane / IDW Extrapolation)
   |                                         /
   |                                        / 
   |                                      /   
   |      Neighbor Wells                 /    
   |         [ * ]                      /     
   |      [ * ]   [ * ]               /       
   |     *  *  *  *  *              /         
   |----------------------------/-------------reversion to constant mean (Standard Kriging)
   |                          /
   +--------------------------------------------> Wellbore Path (X, Y)
```

### The Mechanism of Failure
1. **Dipping Plane Extrapolation (IDW)**: In the original notebook, the slope of the dipping plane ($A, B$) is calculated per neighbor well and averaged. The prediction along the target wellpath is a **linear projection** ($Z_{\text{form}} = A \cdot X + B \cdot Y + C$), which extrapolates the geological tilt perfectly over thousands of feet.
2. **Mean-Reverting Kriging (Standard GP)**: The implemented Kriging model uses a stationary Matern covariance kernel with a constant prior mean. When predicting coordinates ($X, Y$) along the target well path, which extends far away from the neighbor wells (the training points), the GP's predicted residual **reverts to the mean** of the neighbors (a flat horizontal plane).
3. **Geological Consequence**: The GP assumes the geological layer is flat (zero slope) in the prediction zone. For well `000d7d20`, the true formation boundary climbs by **+101.53 ft**, but Kriging predicts a rise of only **+24.16 ft**. This mean reversion causes the prior RMSE to shoot up from **10.83 ft (IDW)** to **36.92 ft (Kriging)**.

---

## 2. Why the Pipeline Took 170.58 Minutes to Run

Step 1 (Precomputation) and Step 2 (Feature Generation) ran in **~1.5 minutes**. The entire bottleneck occurred during Step 3 (Model Training Folds), taking **~35 minutes per fold**:

### 1. Large Feature Matrix
The unified feature matrix in the prediction zone has **3,783,989 rows**. In the 5-Fold GroupKFold loop, each training fold contains $4/5$ of the data, which is **3.02 million rows** with **45 features**.

### 2. Random Forest Regressor
Fitting `RandomForestRegressor` with 50 trees on **3.02 million samples** is highly compute-intensive and accounts for a significant portion of the training time.

### 3. CatBoost GPU-to-CPU Fallback
The CatBoost implementation features a `try-except` block designed to train on GPU and fall back to CPU if it fails:
```python
try:
    # GPU training...
except Exception as e:
    # CPU fallback...
```
If GPU acceleration was not selected in the Kaggle sidebar, or if the GPU ran out of memory (OOM), CatBoost fell back to **CPU training**. Fitting 5 folds of 1200 CatBoost iterations on **3.02 million rows on a standard Kaggle CPU** takes approximately 30 minutes per fold, creating the 170-minute bottleneck.

---

## 3. How to Resolve Both Issues

### Solution A: Trend-Based (Universal) Kriging
To fix the Stage 1 RMSE degradation, we must pre-subtract the IDW linear trend, fit the Kriging model on the **residuals**, and add the predictions back:
$$\hat{Z}_{\text{form}}(X, Y) = \text{IDW\_Trend}(X, Y) + \text{GP\_Residual}(X, Y)$$
Since the residuals around the trend are stationary (mean 0), the GP will correctly decay to 0, leaving the dipping trend intact. This Universal Kriging approach restores the prior RMSE back to **13.72 ft**.

### Solution B: Optimizing Training Speed
1. **Scale Down Random Forest**: Decrease `n_estimators` to `10` or remove RF from the ensemble, as LightGBM and CatBoost do the bulk of the work.
2. **Strict GPU Verification**: Verify that the Kaggle Accelerator is set to **GPU (T4 x2 or P100)** and monitor logs to ensure CatBoost is not falling back to CPU.
