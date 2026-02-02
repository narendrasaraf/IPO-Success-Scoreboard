# Advanced IPO Success Predictor Report

This report describes the implementation of an **Advanced IPO Success Predictor** using machine learning with an emphasis on sophisticated feature engineering, ensemble modeling, and a dynamic prediction system integrated into a Streamlit web application.

---

## Overview and Purpose

The system predicts IPO outcomes classified into three categories:

- **Success (S)**
- **Normal (N)**
- **Fail (F)**

The model leverages financial metrics and market sentiment to classify IPOs with high accuracy, facilitating investors' decisions.

---

## Data Loading and Preprocessing

    # Replace NA values and handle missing data
    df = df.replace(['NA', '', 'NaN', 'null'], 0)
    df = df.fillna(0)
    
    return df
except Exception as e:
    st.error(f"❌ Error loading dataset: {str(e)}")
    return None



- Dataset is loaded and cleaned.
- Missing or malformed values are replaced with zeros.
- Data caching improves app responsiveness.

---

## Advanced Feature Engineering

- Extracts critical financial features, including P/E ratios, ROCE, market capitalization, dividend yield, profits, and sales.
- Creates interaction and polynomial features to represent growth, stability, and valuation risks.
- Encodes target labels for classification.

Notable engineered features include:

- `PE_ROCE_Interaction`
- `Profit_Margin`
- `Market_Cap_to_Sales`
- `Profit_Growth_Momentum`
- `Valuation_Risk`
- `Efficiency_Score`

This enriches the feature space to capture complex relationships.

---

## Ensemble Model Construction

The model ensemble includes:

| Model        | Description                           |
|--------------|-------------------------------------|
| XGBoost      | Gradient boosting tree algorithm    |
| LightGBM     | Fast, gradient boosting framework   |
| CatBoost     | Gradient boosting with categorical handling |
| BalancedRF   | Random forest adjusted for class imbalance |
| GradientBoost| Classic gradient boosting classifier |

- Base models are combined using a **Stacking Classifier** with Logistic Regression as meta-learner.
- A **Voting Classifier** provides a backup ensemble strategy.

---

## Training Pipeline

- Features are scaled using the PowerTransformer (Yeo-Johnson).
- Recursive feature elimination with cross-validation (RFECV) selects the most informative features.
- Class imbalance is addressed with the SMOTETomek approach (oversampling + cleaning).
- Stratified train-test split ensures balanced classes.
- Individual and ensemble models are trained and evaluated with key metrics: accuracy, F1-score, precision, recall.

---

## Prediction System with GMP Integration

- Users input IPO details plus optional **Grey Market Premium (GMP)**.
- The system performs feature engineering and scales input features.
- GMP is incorporated dynamically with a weight dependent on its ratio value, adjusting predictions.
- Confidence calibration considers probability distributions to adjust classification thresholds.
- Outputs include final class prediction, adjusted scores, confidence percentages, and probability breakdowns.

---

## Streamlit Web Application

- Interactive interface supports three modes:

  1. **Model Training:** Load data, engineer features, train ensembles, and save models.
  2. **IPO Prediction:** Input new IPO attributes to receive success predictions and investment recommendations.
  3. **Model Analysis:** Visualize performance metrics and confusion matrices for model evaluation.

- Visualizations include:

  - Bar charts comparing model accuracies and F1-scores.
  - Confusion matrix heatmaps.
  - Color-coded performance scales indicating risk and opportunity.

---

## Conclusion

This solution integrates:

- Domain knowledge through engineered financial features.
- Advanced ensemble learning techniques.
- Dynamic market sentiment incorporation with GMP.
- Robust evaluation with clear visual and textual outputs.

It empowers investors with actionable IPO success predictions supported by rigorous data science and machine learning methodologies.

---

## Appendix: Key Functions

- `load_and_preprocess_data`: Load and clean dataset.
- `advanced_feature_engineering`: Generate enriched features.
- `create_advanced_ensemble`: Build stacking and voting classifiers.
- `train_advanced_model`: Perform feature selection, balancing, training, and saving.
- `predict_with_gmp_advanced`: Predict IPO success with GMP-based score adjustment.
- `display_model_performance`: Visualize and summarize model performances.
- Streamlit UI functions: `main`, `show_model_training`, `show_ipo_prediction`, `show_model_analysis`.

---

_This concludes the report on the Advanced IPO Success Predictor._

---

## 🚀 Deployed App
The application is live on Hugging Face Spaces:
[**IPO Success Predictor**](https://huggingface.co/spaces/NarendraSaraf/IPO-Success-Predictor)

## 💻 How to Run Locally

To run the application on your local machine:

1.  **Navigate to the app directory:**
    ```bash
    cd app2
    ```

2.  **Run the Streamlit app:**
    ```bash
    streamlit run app2.py
    ```

3.  The app will open in your default browser at `http://localhost:8501`.
