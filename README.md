# Term-Life-Insurance-Prediction
 
## Machine Learning Meets Term Life Insurance: Targeting High-Value Customers

###  Author
**Name:** Radadiya Bhavikaben Bavchandbhai  
**Programme:** MSc Data Analytics  
**Year:** 2024  

---

## Project Overview
This project applies **machine learning and data analytics** to identify high-value customers for **term life insurance** at *HashSysTech Insurance*.  
The analysis builds predictive models to optimize marketing efficiency and return on investment.

---

##  Dataset Description
- **Source:** HashSysTech Insurance internal telemarketing dataset  
- **Number of records:** 45,211  
- **Target variable:** `y` (Customer subscribed: yes/no)  

### **Key Features**
| Feature | Type | Description |
|----------|------|-------------|
| Age | Numerical | Age of the customer |
| Work | Categorical | Type of job |
| Marital | Categorical | Marital status |
| education_qual | Categorical | Education qualification |
| call_type | Categorical | Type of contact |
| day | Numerical | Day of the month the call was made |
| Mon | Categorical | Month of call |
| dur | Numerical | Duration of call (seconds) |
| num_calls | Numerical | Number of calls made during campaign |
| prior_outcome | Categorical | Result of previous marketing campaign |

---

##  Data Exploration & Preparation

### 1. **Data Exploration**
- Calculated measures of central tendency and dispersion (mean, median, std)
- Analyzed distributions and feature frequencies

### 2. **Data Cleaning**
- **Missing Values:** Addressed via `dropna()` or `fillna()` (mean/median/mode)
- **Outliers:** Detected with **IQR method** and treated via transformation/capping

### 3. **Data Visualization**
- **Histograms** â€“ visualize age and duration  
- **Boxplots** â€“ detect outliers  
- **Scatter Plots** â€“ explore numerical relationships  

---

##  Machine Learning Models

### 1. **Logistic Regression**
- Binary classification model for conversion likelihood  
- Advantages: Interpretability, efficiency, linear relationships

### 2. **Random Forest**
- Ensemble of decision trees for robust, nonlinear modeling  
- Advantages: Handles nonlinearity, prevents overfitting, interpretable via feature importance

---

### Model Training & Evaluation

### **Data Splitting**
- **Train-Test Split:** 70/30  
- **Stratified Sampling:** Balanced target class distribution  
- Implemented via `train_test_split(stratify=y)`

### **Evaluation Metrics**
| Metric | Description |
|---------|-------------|
| Accuracy | Overall correctness |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | Harmonic mean of Precision & Recall |

---

## Model Performance

| Metric | Logistic Regression | Random Forest |
|---------|--------------------|----------------|
| Accuracy | 0.9004 | 0.8848 |
| Precision | 0.6435 | 0.5485 |
| Recall | 0.3327 | 0.2539 |
| F1-Score | 0.4386 | 0.3471 |

### **Interpretation**
- Logistic Regression performed slightly better overall.  
- Random Forest gave deeper insights into feature importance.  
- Key predictors: **Call Duration**, **Age**, **Num_Calls**, **Prior Outcome**.

---

##  Hyperparameter Tuning
- Used **GridSearchCV** for both models  
- Logistic Regression: increased `max_iter` for convergence  
- Random Forest: tuned `n_estimators`, `max_depth`, `min_samples_split`  
- Logistic Regression improved slightly; Random Forest showed minor overfitting

---

##  Key Findings
- Logistic Regression best suited for this dataset  
- Age and call characteristics drive conversion probability  
- Random Forest less efficient but offered interpretability  
- Stratified sampling improved fairness and model reliability

---

##  Future Work
- Test **Gradient Boosting**, **XGBoost**, **SVM**  
- Enhance feature engineering and normalization  
- Implement **cross-validation** and **dashboard visualizations**

---

##  Tools & Technologies
| Category | Tools |
|-----------|--------|
| Language | Python 3.x |
| Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn |
| Environment | Jupyter Notebook / Google Colab |
| Version Control | Git & GitHub |

---

##  Folder Structure
```
fundamentals-of-data-analytics/
‚
 data/                       # Dataset
 notebook/                  # Colab notebook
 images/                    # Reports and visuals
 requirements.txt            # Dependencies
 README.md                   # Documentation
```

---

## References
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5â€“32.  
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.  
- James, G. et al. (2013). *An Introduction to Statistical Learning with Applications in R*. Springer.  
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.  
- Powers, D.M.W. (2011). *Evaluation: From Precision, Recall and F-measure to ROC*. JMLT.  
- Munzner, T. (2014). *Visualization Analysis and Design*. CRC Press.

---

##  Conclusion
This project completes the **data analytics lifecycle**, from exploration to model evaluation.  
By integrating statistical methods with machine learning, it reveals key predictors influencing life-insurance conversions.  
The analysis supports **data-driven marketing** and improved **customer targeting** strategies.

---

Â© 2024 Bhavikaben Radadiya Â· MSc Data Analytics


