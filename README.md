# Titanic Survival Prediction (Logistic Regression)

Binary classification project analyzing Titanic passenger survival using an interpretable logistic regression model, with a focus on clean machine learning workflow, generalization, and avoiding overfitting.

---

## Project Overview

This project solves a binary classification problem: predicting whether a passenger survived the Titanic disaster.
The goal was **not leaderboard optimization**, but building a **clear, interpretable, and well-reasoned ML pipeline** with strong generalization and disciplined model selection.

---

## Problem Type

- **Task**: Binary classification  
- **Target**: `Survived`
  - `0` — Did not survive  
  - `1` — Survived  

---

## Dataset

- **Size**: < 1000 samples  
- **Feature types**: Numerical + categorical  
- **Target distribution**: Moderately imbalanced (~60% / 40%)

### Missing values
- `Age`: ~20%
- `Cabin`: ~77%
- `Embarked`: ~0.2%

---

## Key Constraints

- Interpretability over complexity  
- Avoid overfitting  
- No focus on training or inference speed  

These constraints guided all modeling decisions.

---

## Key Insights (EDA)

- **Sex** and **passenger class** are strong main effects with a direct impact on survival.
- **Cabin missingness** strongly correlates with lower passenger class and lower survival probability.
- **Age missingness** does not show predictive signal and appears largely random.
- **Embarked** reflects passenger composition and ticket price differences rather than a causal effect.
- The dataset is small and dominated by **additive effects**, favoring simpler linear models over complex architectures.

---

## Approach

1. Exploratory Data Analysis (EDA)
2. Missing value analysis and handling
3. Feature engineering (Deck extraction, Title extraction)
4. Model selection based on data properties and constraints
5. Logistic Regression as the final model
6. Validation using appropriate metrics

---

## Model Choice

**Logistic Regression** was selected because:

- The dataset is small
- Strong main effects dominate
- Relationships are largely monotonic and additive
- Interpretability is a priority
- The model is stable and resistant to overfitting

More complex models were intentionally avoided due to diminishing returns.

---

## Evaluation Strategy

- **Primary metric**: F1-score  
  (chosen due to moderate class imbalance)
- **Secondary metric**: Accuracy  
  (used as a sanity check)

Validation performance was used for model selection before final training.

---

## Results

- **Validation F1-score**: ~0.77  
- **Public Kaggle score**: **0.77** (accuracy, rounded from 0.76794)

The public leaderboard score closely matches validation performance, indicating good generalization and no data leakage.

---

## Conclusion

Logistic regression provided a stable, interpretable, and well-generalizing solution for this dataset.
Further model complexity was intentionally avoided, as expected gains were marginal relative to the increased risk of overfitting and reduced interpretability.

---

## How to Run

See the Reproducibility section below for full setup instructions.

---

## Reproducibility

To reproduce the results:

1. Download `train.csv` and `test.csv` from the Kaggle Titanic competition:  
   https://www.kaggle.com/competitions/titanic

2. Place the files in the `data/` directory:

```
data/
├── train.csv
└── test.csv
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run training and generate the submission:
```
python src/train.py
```
The script will train the model and generate a submission file:
```
submission.csv
```