# Titanic Survival Prediction
Machine learning project to predict passenger survival using the 
Kaggle Titanic dataset. Focused on production-style preprocessing 
pipelines and proper model evaluation.

## Results
- Cross-validated accuracy: **77%** (StratifiedKFold, 5 folds)
- Evaluation: cross_val_score with StratifiedKFold to preserve 
  class balance across folds

## Pipeline architecture
Full scikit-learn Pipeline to prevent data leakage:

- **StandardScaler** → Age (normally distributed)
- **RobustScaler** → SibSp, Parch, Fare (outlier-resistant)
- **OneHotEncoder** → Pclass, Sex, Embarked, Title (extracted 
  from Name)
- **LDA** → dimensionality reduction before classification
- **Logistic Regression** → final classifier (saga solver)

## Feature engineering
- Extracted passenger Title from Name column as a new categorical 
  feature using a custom FunctionTransformer

## Stack
Python · scikit-learn · pandas · Pipeline · ColumnTransformer · 
StratifiedKFold

## How to run
1. Open the notebook in Google Colab using the link above
2. Upload train.csv and test.csv from Kaggle
3. Run all cells in order
