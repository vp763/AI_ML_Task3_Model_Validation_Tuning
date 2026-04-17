# AI & ML Internship — Task 3: Model Validation, Overfitting Control & Hyperparameter Tuning

This is Task 3 of the Maincrafts Technology AI/ML internship. The goal here was to go beyond just training models and actually make them reliable — detecting overfitting, validating with cross-validation, and tuning hyperparameters properly.

---

## What This Task Covers

Building on the California Housing dataset from Task 2, this task adds:

- Detecting overfitting by comparing train vs test performance
- Cross-validation to get stable, realistic performance estimates
- Hyperparameter tuning using GridSearchCV
- Final model selection with proper justification

---

## Dataset

**California Housing Dataset** (from `sklearn.datasets`)

- 20,640 samples
- 8 input features: median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, longitude
- Target: Median house value

---

## Project Structure

```
├── AI_ML_Task3_Model_Validation_Tuning.ipynb   # Main notebook
├── overfitting_analysis.png                    # Train vs test RMSE plot
├── model_comparison.png                        # Final model comparison chart
├── best_model.pkl                              # Saved best model (joblib)
├── scaler.pkl                                  # Saved StandardScaler
├── Task3_Report.pdf                            # Analysis report (2-3 pages)
└── README.md
```

---

## Steps Implemented

**Step 1-4:** Import libraries, load dataset, scale features, train-test split

**Step 5:** Overfitting detection — trained an unconstrained Decision Tree and compared train RMSE vs test RMSE. The large gap confirmed overfitting.

**Step 6:** 5-Fold Cross-Validation on all three models (Linear Regression, Ridge, Decision Tree)

**Step 7:** GridSearchCV to tune:
- Decision Tree: `max_depth` ∈ {3, 5, 7, 10}, `min_samples_split` ∈ {2, 5, 10}
- Ridge: `alpha` ∈ {0.01, 0.1, 1.0, 10.0, 100.0}

**Step 8-9:** Evaluated optimized models on the held-out test set and built a comparison table

**Step 10:** Final model selection with written justification

---

## Results Summary

| Model | Train RMSE | Test RMSE | R² Score |
|-------|-----------|-----------|----------|
| Linear Regression | ~0.72 | ~0.73 | ~0.60 |
| Ridge (Tuned) | ~0.72 | ~0.73 | ~0.60 |
| Decision Tree (Tuned) | ~0.55 | ~0.63 | ~0.72 |

> Actual values will appear after running the notebook — these are approximate.

---

## Key Observations

- The unconstrained Decision Tree had near-zero train RMSE but much higher test RMSE — classic overfitting
- After tuning `max_depth` and `min_samples_split`, the gap reduced significantly
- Cross-validation scores were more stable than single-split results
- Ridge and Linear Regression performed similarly — the dataset has some non-linearity that linear models miss

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/ai-ml-task3.git
cd ai-ml-task3

# Install dependencies
pip install scikit-learn pandas numpy matplotlib joblib jupyter

# Launch notebook
jupyter notebook AI_ML_Task3_Model_Validation_Tuning.ipynb
```

---

## Tools Used

- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas, NumPy
- matplotlib
- joblib

---

## Internship

Maincrafts Technology — Artificial Intelligence & Machine Learning Internship  
Task 3 of the internship series
