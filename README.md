# Stress-Detection

## Analysis Workflow
1. Preprocessing
2. Exploratory Data Analysis
3. Statistical Hypothesis Test
4. Factor Analysis
5. Principal Component Analysis
6. Modeling
7. 95% Confidence Interval Visualization

### Overview
This project explores whether stress levels can be quantitatively predicted using only self-reported survey data, without expensive wearable devices or physiological sensors.
Using survey responses from undergraduate students at Korea University (Sejong Campus), we build and evaluate machine learning models to predict perceived stress scores.

### Research Question
Can we build a reliable AI model that predicts stress levels using only self-reported questionnaire data?

## Dataset
- Participants: Undergraduate students (Korea University Sejong Campus)
- Format: Online self-report survey
- Sample Size: 73 participants
- Features: 73 survey variables
- Target Variable: PSS-10 (Perceived Stress Scale) — mean score

## Statistical Analysis
Before modeling, statistical validation and exploratory analysis were conducted.

### Reliability Test
- Cronbach’s Alpha

### Group Difference Tests
- Independent t-test
- ANOVA

### Factor Suitability Checks
- KMO Test
- Bartlett’s Test of Sphericity

## Dimensionality Reduction
To address multicollinearity and small sample size:

### Factor Analysis
- Varimax rotation
- 18 latent factors extracted

### Principal Component Analysis (PCA)
- Based on extracted factors
- Reduced to 8 principal components
- Cumulative explained variance: 82%

## Modeling
The following regression models were implemented:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Extreme Gradient Boosting (XGBoost)

## Model Stabilization
To improve robustness given the small dataset:
- Bootstrap sampling (1,000 iterations)

Estimated:
- Mean Squared Error (MSE)
- 95% Confidence Interval
