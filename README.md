# Diabetes Prediction 

A complete end-to-end machine learning project that predicts whether a patient 
has diabetes using the Pima Indians Diabetes Dataset. The project walks through 
every stage of a real ML workflow: exploratory data analysis, careful preprocessing 
of a medically noisy dataset, training and comparing 6 classification models, and 
a full evaluation suite focused on clinically meaningful metrics.

Built with Python, scikit-learn, and XGBoost.

## Highlights

- Identified and handled **hidden missing values** — five features used 0 as a 
  placeholder for missing data, which is biologically impossible (e.g. BMI = 0). 
  Imputed using class-wise median to avoid bias between diabetic and non-diabetic groups.

- Applied **Winsorizing** (1st–99th percentile capping) instead of dropping rows, 
  preserving all 768 samples while reducing the influence of extreme values like 
  Insulin = 846 μU/mL.

- Enforced **no data leakage** — the train/test split always happens before 
  scaling. The StandardScaler is fit only on training data, then applied to the 
  test set with the same parameters.

- Evaluated using **5 metrics** — not just accuracy. In a medical screening tool, 
  a model that misses diabetic patients is dangerous regardless of its accuracy score. 
  Priority order: Recall → F1-Score → ROC-AUC → Precision → Accuracy.

- Compared **6 models** on identical data splits: Logistic Regression, KNN, SVM, 
  Decision Tree, Random Forest, and XGBoost — with 5-fold cross-validation to 
  detect overfitting.

  ## Dataset

**Pima Indians Diabetes Dataset** — National Institute of Diabetes and Digestive 
and Kidney Diseases.

- 768 female patients, all aged ≥ 21, of Pima Indian heritage
- 8 clinical features, 1 binary target (Outcome: 0 = No Diabetes, 1 = Diabetes)
- Class balance: 65% non-diabetic / 35% diabetic (mild imbalance)

| Feature | Description | Unit |
|---------|-------------|------|
| Glucose | Plasma glucose — 2-hour oral glucose tolerance test | mg/dL |
| BloodPressure | Diastolic blood pressure | mm Hg |
| SkinThickness | Triceps skinfold thickness | mm |
| Insulin | 2-hour serum insulin | μU/mL |
| BMI | Body Mass Index | kg/m² |
| DiabetesPedigreeFunction | Genetic diabetes risk score | score |
| Age | Age | years |
| Pregnancies | Number of pregnancies | count |

## Approach

### 1. Exploratory Data Analysis
Visualised feature distributions split by outcome class, computed a correlation 
matrix, and identified the extent of hidden missing values across all features.
Key finding: Glucose is the strongest single predictor (r = 0.47), with diabetic 
patients averaging 35 mg/dL higher than non-diabetic patients on the 2-hour OGTT.

### 2. Preprocessing
Five steps applied in strict order:
1. **Zero imputation** — replaced biologically impossible zeros with the class-wise 
   median (grouped by Outcome) to avoid polluting either group's statistics.
2. **Outlier capping** — clipped all features to the 1st–99th percentile range 
   (Winsorizing), reducing Insulin's max from 846 to ~520 without losing any rows.
3. **Stratified split** — 80/20 train/test split with stratify=y to preserve the 
   65/35 class ratio in both sets.
4. **Feature scaling** — StandardScaler fit on training data only, then applied to 
   the test set to prevent leakage.
5. **Class imbalance** — handled via class_weight='balanced' in all sklearn models 
   and scale_pos_weight in XGBoost (value = 1.87 = 400/214).

### 3. Model Training
Six models trained on the same data with a fixed random seed (42) for full 
reproducibility. Models that require distance or magnitude calculations (Logistic 
Regression, KNN, SVM) receive scaled features; tree-based models receive raw features.

### 4. Evaluation
Each model is evaluated on the held-out test set across five metrics and also 
assessed via 5-fold stratified cross-validation on the training set. A large gap 
between test F1 and CV F1 signals overfitting.

## Results

| Model | Recall | F1-Score | ROC-AUC | Accuracy |
|-------|:------:|:--------:|:-------:|:--------:|
| Logistic Regression | — | — | — | — |
| KNN | — | — | — | — |
| SVM | — | — | — | — |
| Decision Tree | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

> Metric priority: **Recall** is most important in a medical screening context — 
> missing a diabetic patient is a more serious error than a false alarm.


