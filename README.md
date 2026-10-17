# 🧠 Stroke Risk Predictor

**An Interactive Machine Learning Application for Exploring Stroke Classification**

Python | Scikit-learn | XGBoost | Streamlit

## Project Overview

Stroke Risk Predictor is an interactive web application that demonstrates how machine learning can be integrated into a digital health interface.

Built with Python and Streamlit, the application accepts demographic, health and lifestyle information, processes these inputs using a saved preprocessing object and classification model, and displays a stroke-class prediction alongside the model's estimated probability.

The project brings together machine-learning experimentation, healthcare data analysis, model interpretation and interactive application development.

**Project status:** Educational machine-learning prototype. The application is not clinically validated and should not be used to assess an individual's medical risk.

## Project Objectives

- Explore stroke classification using demographic and health-related data.
- Develop and evaluate machine-learning models using a public healthcare dataset.
- Integrate a saved model into an interactive web application.
- Present classification outputs through accessible visualisations.
- Explore global feature importance and predefined health-risk indicators.
- Examine the limitations of predictive models in healthcare applications.

## Dataset

The project uses a publicly available **Stroke Prediction Dataset from Kaggle**.

The dataset includes demographic, clinical and lifestyle information alongside a recorded stroke outcome used for binary classification.

The application's input features include:

| Category | Variables |
|---|---|
| Demographics | Gender, age, marital history, residence type |
| Health indicators | Hypertension, heart disease, average glucose level, BMI |
| Socioeconomic information | Work type |
| Lifestyle | Smoking status |

The dataset version, sample size, class distribution and detailed preprocessing decisions are documented or explored in the project notebooks.

## Repository Structure

```text
Stroke-Risk-prediction-App/
├── app.py
├── requirements.txt
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── modelling.ipynb
└── models/
    ├── stroke_model.pkl
    └── scaler.pkl
```

The structure above reflects the supplied notebook names and model paths referenced by the application. The saved model files must be present for the application to run.

## Machine Learning Workflow

### 1. Exploratory Data Analysis

The `eda.ipynb` notebook is dedicated to exploring the dataset and understanding the characteristics of the variables used for classification.

### 2. Data Preprocessing

The `preprocessing.ipynb` notebook contains the data preparation work supporting model development.

The deployed application loads a saved preprocessing scaler from `models/scaler.pkl` and applies it to the input data before prediction.

### 3. Model Development

The `modelling.ipynb` notebook contains the machine-learning experimentation.

The project's Python dependencies include Scikit-learn, XGBoost and Imbalanced-learn.

The application loads its trained model from `models/stroke_model.pkl`. The exact final estimator, evaluation metrics and training configuration should be confirmed against the modelling notebook before reporting results.

### 4. Model Integration

The trained model and preprocessing object are loaded using Joblib.

When a user submits the form, the application:

1. Collects demographic, health and lifestyle inputs.
2. Converts categorical responses into numerical features.
3. Constructs a feature DataFrame.
4. Applies the saved scaler.
5. Generates a model probability using `predict_proba()`.
6. Applies a classification threshold of **0.30**.
7. Displays the classification and supporting visualisations.

## Application Features

### Interactive Data Entry

The interface organises inputs into three sections:

**Demographics:** Gender, age, marital history and residential setting.

**Health Conditions:** Hypertension, heart disease, average glucose level and BMI.

**Lifestyle:** Employment type and smoking status.

### Classification Output

The application displays a *High Risk* or *Low Risk* label based on whether the model's output probability reaches the configured 0.30 threshold.

These are application-defined classification labels, not clinically established risk categories.

### Probability Visualisation

A percentage display and horizontal risk gauge visualise the model output and indicate the classification threshold.

The displayed probability should not be interpreted as a validated estimate of an individual's future stroke risk.

### Global Feature Importance

The application presents a bar chart of the ten highest feature-importance scores reported by the saved model.

These scores describe the model's overall use of features. They do not explain the contribution of each feature to an individual prediction.

### Health-Risk Flags

The interface also displays rule-based indicators for selected characteristics, including:

- Age of 60 years or above
- Reported hypertension
- Reported heart disease
- Average glucose level above 140
- BMI above 30
- Current or previous smoking

These indicators are derived from predefined application rules and should not be confused with individually calculated feature contributions from the machine-learning model.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application and modelling language |
| Pandas and NumPy | Data manipulation |
| Scikit-learn | Machine-learning and preprocessing utilities |
| XGBoost | Machine-learning experimentation |
| Imbalanced-learn | Tools for imbalanced classification |
| Joblib | Saving and loading trained objects |
| Streamlit | Interactive application interface |
| Matplotlib | Data visualisation |
| Seaborn | Exploratory visualisation support |

## Installation

### Clone the repository

```bash
git clone https://github.com/Amara-4312/Stroke-Risk-prediction-App.git
cd Stroke-Risk-prediction-App
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

The application requires the saved model and scaler files referenced in `app.py`.

### Model Development and Deployment

The application uses an **XGBoost Classifier (`XGBClassifier`)** trained for binary stroke classification.

The trained estimator is saved using Joblib and loaded into a Streamlit application alongside a separate preprocessing scaler.

During prediction, the application transforms user-provided health and demographic information, obtains the model's estimated probability for the positive stroke class, and applies a classification threshold of 0.30.

The interface also displays global feature-importance scores from the trained model.

The model remains an exploratory research prototype and has not been clinically validated.

## Model Evaluation

The modelling notebook contains the experiments used during model development.

Before publishing quantitative model-performance claims, the final estimator and its results should be verified.

Relevant metrics include:

- Precision and recall for the stroke-positive class
- F1-score
- Confusion matrix
- ROC-AUC and precision–recall analysis
- Performance under the selected classification threshold

Particular attention should be given to class imbalance and the consequences of false-positive and false-negative classifications.

## Limitations

**Clinical validation:** The model has not been validated for clinical diagnosis, screening or individual medical decisions.

**Dataset generalisability:** A publicly available stroke dataset may not represent patients across different healthcare systems or population groups.

**Model probabilities:** The interface presents model-generated probabilities, but their calibration and clinical interpretation have not been established.

**Classification threshold:** The application uses a threshold of 0.30. Its appropriateness requires validation against the intended objective and independent evaluation data.

**Feature importance:** Global importance scores do not represent individual-level causal explanations.

**Risk flags:** The displayed flags use predefined application rules rather than a personalised clinical risk-assessment framework.

**Preprocessing consistency:** The order and definitions of application inputs must match the data and transformations used when training the saved model.

## Future Improvements

- Verify and document final model evaluation metrics.
- Evaluate probability calibration and classification-threshold selection.
- Assess model performance across demographic subgroups.
- Strengthen reproducibility of the training and preprocessing workflow.
- Explore appropriate methods for individual prediction explanation.
- Improve the presentation of uncertainty and model limitations.
- Conduct external validation before considering any clinical application.

## Research and Technical Significance

This project demonstrates the process of connecting supervised machine-learning workflows to an interactive application.

It provided practical experience in model integration, input processing, prediction visualisation and communicating the limitations of health-related machine-learning outputs.

The project also illustrates an important distinction in digital health research: building a functional predictive prototype is different from establishing a reliable clinical decision-support system.

## Author

**Amarachi Emmanuella Nnanyelu**

Applied Biochemistry graduate interested in computational biology, health data science, interdisciplinary biomedical research and artificial intelligence.

[GitHub](https://github.com/Amara-4312) | [LinkedIn](https://www.linkedin.com/in/amarachinnanyelu)

---

**Medical Disclaimer:** This application is intended solely for educational and exploratory research purposes. It is not a diagnostic tool or clinically validated risk calculator. Its outputs should not guide medical decisions.
