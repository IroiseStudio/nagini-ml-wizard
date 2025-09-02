---
license: cc
title: Nagini_ML_Wizard
sdk: gradio
colorFrom: purple
colorTo: pink
short_description: Nagini ML Wizard guides full ML workflow
sdk_version: 5.42.0
---

# 🐍🧙 Nagini ML Wizard

**Nagini ML Wizard** is an interactive, step-by-step Gradio app that guides you through the complete machine learning workflow.

---

## ✨ What It Does

Nagini ML Wizard walks you through all the essential stages of a machine learning project:

1. **Data Import**

   - Upload a CSV or load sample datasets (Iris, Wine, Diabetes, etc.)
   - Select your target column and choose which features to include

2. **Preprocessing**

   - Handle missing values (mean, median, mode, or drop rows)
   - Scale numeric features, encode categoricals
   - Guardrails to prevent target leakage

3. **Exploratory Data Analysis (EDA)**

   - Visualize target distribution, class balance, correlations, and scatter plots
   - Download plots as PNGs for reports

4. **Model Training**

   - Pick task type: **classification** or **regression**
   - Choose from popular models: Decision Tree, Random Forest, KNN, Logistic/Linear Regression, Naive Bayes, SVM, Multilayer Perceptron
   - Configure hyperparameters with sensible defaults

5. **Evaluation**

   - Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
   - Regression: MAE, RMSE, R², residual plots
   - Feature importances and permutation importance for interpretability

6. **Prediction**

   - Try single-row inputs with auto-generated forms
   - Upload a CSV for batch predictions
   - View probabilities (for classifiers) or numeric estimates (for regressors)

7. **Export**
   - Download the trained pipeline as a `.joblib` file
   - Reuse the model for inference later

---

## 🚀 How to Run

### Run on Hugging Face Spaces

Nagini ML Wizard is also deployed as a free interactive demo on [Hugging Face Spaces](https://huggingface.co/spaces/AlbanDelamarre/Nagini_ML_Wizard)
Every push to the GitHub repo triggers CI/CD via GitHub Actions to redeploy the Space automatically.

### Run locally (Windows + VS Code)

```powershell
# Clone the repo
git clone https://github.com/IroiseStudio/Nagini-ML-Wizard.git
cd Nagini-ML-Wizard

# Create and activate a virtual environment
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch the wizard
python app.py
```

## Author

**Alban Delamarre**  
[Hugging Face Spaces](https://huggingface.co/AlbanDelamarre)
