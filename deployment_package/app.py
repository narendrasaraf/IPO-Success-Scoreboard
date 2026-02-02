
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Set page configuration
st.set_page_config(
    page_title="Advanced IPO Success Predictor",
    page_icon="📈",
    layout="wide"
)


# ============================================================================
# MODEL LOADING SECTION (Replaces Training)
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_artifact(path):
    """Load a single pickle artifact"""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        raise

@st.cache_resource(show_spinner=True)
def load_all_models():
    """Load all required pre-trained artifacts"""
    artifacts = {}
    
    try:
        # Load all model artifacts
        artifacts['model'] = load_artifact("advanced_ensemble_model.pkl")
        artifacts['scaler'] = load_artifact("advanced_scaler.pkl")
        artifacts['selector'] = load_artifact("advanced_selector.pkl")
        artifacts['selected_features'] = load_artifact("selected_features.pkl")
        artifacts['label_encoder'] = load_artifact("label_encoder.pkl")
        
        st.success("✅ All model artifacts loaded successfully!")
        return artifacts
        
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please ensure all .pkl files are in the same directory as this script.")
        raise
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        raise

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_feature_engineering(df):
    """Apply the same feature engineering used during training"""
    df = df.copy()
    
    # Define core columns
    core_cols = [
        'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
        'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
        'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
        'Issue Price (Rs)', 'ROCE %'
    ]
    
    # Ensure all columns exist and are numeric
    for col in core_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Advanced Feature Engineering (Must match training exactly)
    df['PE_ROCE_Interaction'] = df['P/E'] * df['ROCE %']
    df['Profit_Margin'] = (df['Net Profit of last quarter Rs. Cr.'] / (df['Quarterly Sales Rs.Cr.'] + 1)) * 100
    df['Market_Cap_to_Sales'] = df['Mar Capitalization Rs.Cr.'] / (df['Quarterly Sales Rs.Cr.'] + 1)
    df['Value_Growth_Score'] = (df['Dividend Yield %'] * 0.3) + (df['ROCE %'] * 0.7)
    
    df['Profit_Growth_Momentum'] = df['Quarterly Profit Variation %'] * np.log1p(np.abs(df['Net Profit of last quarter Rs. Cr.']))
    df['Sales_Growth_Momentum'] = df['Quarterly Sales Variation %'] * np.log1p(df['Quarterly Sales Rs.Cr.'])
    df['Composite_Growth_Score'] = (df['Quarterly Profit Variation %'] + df['Quarterly Sales Variation %']) * df['ROCE %']
    
    df['Profit_Stability'] = 1 / (1 + np.abs(df['Quarterly Profit Variation %']))
    df['Size_Stability'] = np.log1p(df['Mar Capitalization Rs.Cr.'])
    df['Valuation_Risk'] = df['P/E'] / (df['ROCE %'] + 1)
    
    df['PE_Squared'] = df['P/E'] ** 2
    df['ROCE_Squared'] = df['ROCE %'] ** 2
    df['Profit_Var_Squared'] = df['Quarterly Profit Variation %'] ** 2
    df['Size_ROCE_Interaction'] = df['Mar Capitalization Rs.Cr.'] * df['ROCE %']
    
    df['Efficiency_Score'] = (df['ROCE %'] * df['Profit_Margin']) / (np.abs(df['P/E']) + 1)
    df['Growth_Quality'] = (df['Quarterly Profit Variation %'] * df['Profit_Margin']) / 100
    df['Market_Sentiment'] = (df['P/E'] * df['Dividend Yield %']) / 100
    
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def calculate_dynamic_gmp_weight(gmp_ratio: float) -> float:
    """Calculate dynamic weight for GMP contribution"""
    if gmp_ratio <= 20:
        return 0.1
    elif gmp_ratio <= 50:
        return 0.2
    elif gmp_ratio <= 100:
        return 0.3
    else:
        return 0.25

def calculate_prediction_confidence(probabilities):
    """Calculate prediction confidence based on probability distribution"""
    sorted_probs = sorted(probabilities, reverse=True)
    if len(sorted_probs) >= 2:
        confidence = sorted_probs[0] - sorted_probs[1]
    else:
        confidence = sorted_probs[0]
    return min(confidence * 1.5, 1.0)

def determine_final_classification(adjusted_score, confidence):
    """Determine final class based on score and confidence"""
    confidence_adjustment = (1 - confidence) * 10
    
    if adjusted_score >= (75 - confidence_adjustment):
        return 'S'
    elif adjusted_score >= (40 - confidence_adjustment):
        return 'N'
    else:
        return 'F'

def predict_with_gmp_advanced(model, scaler, selector, input_features: dict, 
                             gmp_ratio: float, selected_features, label_encoder):
    """Run full prediction pipeline using preloaded artifacts"""
    try:
        input_df = pd.DataFrame([input_features])
        input_df = apply_feature_engineering(input_df)
        
        scaled = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled, columns=input_df.columns)
        
        selected_array = selector.transform(scaled_df)
        input_final = pd.DataFrame(selected_array, columns=selected_features)
        
        proba = model.predict_proba(input_final)[0]
        pred_encoded = int(model.predict(input_final)[0])
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        
        classes = list(label_encoder.classes_)
        if 'S' in classes:
            success_idx = classes.index('S')
            base_success_prob = proba[success_idx] * 100
        else:
            base_success_prob = max(proba) * 100
        
        if gmp_ratio > 0:
            gmp_weight = calculate_dynamic_gmp_weight(gmp_ratio)
            gmp_contribution = gmp_weight * gmp_ratio
            adjusted_score = min(base_success_prob + gmp_contribution, 95.0)
        else:
            adjusted_score = base_success_prob
            gmp_contribution = 0.0
        
        confidence = calculate_prediction_confidence(proba)
        final_class = determine_final_classification(adjusted_score, confidence)
        
        prob_dict = dict(zip(classes, proba))
        
        return {
            'predicted_class': final_class,
            'base_score': base_success_prob,
            'adjusted_score': adjusted_score,
            'gmp_contribution': gmp_contribution,
            'confidence': confidence * 100,
            'probabilities': prob_dict
        }
        
    except Exception as e:
        st.error(f"❌ Prediction pipeline error: {e}")
        return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🎯 Advanced IPO Success Prediction Platform")
    st.markdown("---")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model_performance' not in st.session_state:
        st.session_state.model_performance = None
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Model Training", "IPO Prediction", "Model Analysis"]
    )
    
    if app_mode == "Model Training":
        show_model_training()
    elif app_mode == "IPO Prediction":
        show_ipo_prediction()
    else:
        show_model_analysis()

def show_model_training():
    st.header("🔬 Advanced Model Training")
    
    st.info("""
    **⚠️ Training section has been replaced with model loading.**
    
    This production version loads pre-trained models from .pkl files instead of training.
    
    Please ensure you have the following files:
    - advanced_ensemble_model.pkl
    - advanced_scaler.pkl
    - advanced_selector.pkl
    - selected_features.pkl
    - label_encoder.pkl
    """)
    
    if st.button("🔄 Load Pre-trained Models"):
        try:
            with st.spinner("Loading model artifacts..."):
                artifacts = load_all_models()
            
            st.session_state.model_trained = True
            st.success("✅ All models loaded successfully!")
            
            # Display model info
            st.subheader("📊 Model Information")
            st.write(f"- **Selected Features**: {len(artifacts['selected_features'])} features")
            st.write(f"- **Model Classes**: {list(artifacts['label_encoder'].classes_)}")
            st.write("- **Model Type**: Advanced Ensemble")
            st.write("- **Preprocessing**: PowerTransformer + RFECV")
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")

def show_ipo_prediction():
    st.header("🔮 IPO Success Prediction")
    
    # Load trained models and encoders
    try:
        artifacts = load_all_models()
    except Exception:
        st.error("❌ Models not found. Please load the models first in 'Model Training' tab.")
        return
    
    st.info("""
    **🎯 Classification System:**
    - **Success (S)**: Strong fundamentals + positive market sentiment
    - **Normal (N)**: Moderate potential with acceptable risk
    - **Fail (F)**: High risk or weak fundamentals
    
    **📊 GMP Integration:** Dynamically weighted based on market conditions
    """)
    
    # GMP Input First
    st.subheader("📈 Step 1: Enter Grey Market Premium (GMP)")
    gmp_ratio = st.number_input(
        "GMP Ratio",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
        help="Enter the Grey Market Premium ratio (0 if not available)"
    )
    
    if gmp_ratio > 0:
        st.success(f"✅ GMP Ratio: {gmp_ratio} (Will contribute to prediction)")
    
    # IPO Details Form
    with st.form("advanced_ipo_prediction"):
        st.subheader("📋 Step 2: Enter IPO Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pe_ratio = st.number_input("P/E Ratio", min_value=0.0, value=20.0, step=1.0)
            market_cap = st.number_input("Market Capitalization (Rs. Cr.)", min_value=0.0, value=1500.0, step=100.0)
            dividend_yield = st.number_input("Dividend Yield %", min_value=0.0, value=2.5, step=0.1)
        
        with col2:
            net_profit = st.number_input("Net Profit last quarter (Rs. Cr.)", min_value=0.0, value=75.0, step=10.0)
            profit_variation = st.number_input("Quarterly Profit Variation %", value=15.0, step=1.0)
            quarterly_sales = st.number_input("Quarterly Sales (Rs. Cr.)", min_value=0.0, value=600.0, step=50.0)
        
        with col3:
            sales_variation = st.number_input("Quarterly Sales Variation %", value=12.0, step=1.0)
            issue_price = st.number_input("Issue Price (Rs)", min_value=0.0, value=120.0, step=10.0)
            roce = st.number_input("ROCE %", value=18.0, step=1.0)
        
        submitted = st.form_submit_button("🎯 Predict IPO Success")
    
    if submitted:
        input_features = {
            'P/E': pe_ratio,
            'Mar Capitalization Rs.Cr.': market_cap,
            'Dividend Yield %': dividend_yield,
            'Net Profit of last quarter Rs. Cr.': net_profit,
            'Quarterly Profit Variation %': profit_variation,
            'Quarterly Sales Rs.Cr.': quarterly_sales,
            'Quarterly Sales Variation %': sales_variation,
            'Issue Price (Rs)': issue_price,
            'ROCE %': roce
        }
        
        with st.spinner("🔄 Performing advanced analysis..."):
            result = predict_with_gmp_advanced(
                artifacts['model'], 
                artifacts['scaler'], 
                artifacts['selector'], 
                input_features, 
                gmp_ratio, 
                artifacts['selected_features'], 
                artifacts['label_encoder']
            )
        
        if result:
            display_advanced_results(result, gmp_ratio)

def display_advanced_results(result, gmp_ratio):
    """Display advanced prediction results"""
    
    st.subheader("🎯 Prediction Results")
    
    # Color coding based on prediction
    if result['predicted_class'] == 'S':
        color = "green"
        icon = "✅"
        message = "HIGH SUCCESS POTENTIAL"
    elif result['predicted_class'] == 'N':
        color = "blue"
        icon = "ℹ️"
        message = "MODERATE POTENTIAL"
    else:
        color = "red"
        icon = "❌"
        message = "HIGH RISK"
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Classification", f"{icon} {result['predicted_class']}")
    
    with col2:
        st.metric("Adjusted Score", f"{result['adjusted_score']:.1f}")
    
    with col3:
        st.metric("Confidence", f"{result['confidence']:.1f}%")
    
    with col4:
        st.metric("Base Score", f"{result['base_score']:.1f}")
    
    # Detailed breakdown
    st.subheader("📊 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Score Breakdown:**")
        st.write(f"- Base Model Score: {result['base_score']:.1f}")
        if gmp_ratio > 0:
            st.write(f"- GMP Contribution: +{result['gmp_contribution']:.1f}")
        st.write(f"- **Final Score: {result['adjusted_score']:.1f}**")
        
        st.write("**Probability Distribution:**")
        for cls, prob in result['probabilities'].items():
            st.write(f"- {cls}: {prob*100:.1f}%")
    
    with col2:
        st.write("**Classification Thresholds:**")
        st.write("- Success (S): ≥ 70")
        st.write("- Normal (N): 40 - 69")
        st.write("- Fail (F): < 40")
        st.write(f"- **Your Score: {result['adjusted_score']:.1f}**")
    
    # Visual scale
    st.subheader("📈 Performance Scale")
    create_visual_scale(result['adjusted_score'])
    
    # Recommendation
    st.subheader("💡 Investment Recommendation")
    display_recommendation(result, gmp_ratio)

def create_visual_scale(score):
    """Create a visual performance scale"""
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create gradient background
    x = np.linspace(0, 100, 100)
    colors = ['red'] * 40 + ['blue'] * 30 + ['green'] * 30
    
    for i in range(len(x)-1):
        ax.axvspan(x[i], x[i+1], color=colors[i], alpha=0.3)
    
    # Add score marker
    ax.axvline(x=score, color='black', linewidth=3, label=f'Score: {score:.1f}')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Performance Score')
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add threshold labels
    ax.text(20, 0.5, 'FAIL', ha='center', va='center', fontweight='bold')
    ax.text(55, 0.5, 'NORMAL', ha='center', va='center', fontweight='bold')
    ax.text(85, 0.5, 'SUCCESS', ha='center', va='center', fontweight='bold')
    
    st.pyplot(fig)
    plt.close()

def display_recommendation(result, gmp_ratio):
    """Display investment recommendation"""
    
    if result['predicted_class'] == 'S':
        st.success("""
        **🎉 STRONG BUY RECOMMENDATION**
        
        **Key Strengths:**
        - Strong fundamental metrics
        - Positive growth indicators
        - Favorable risk-reward ratio
        
        **Action:** Consider applying with high allocation. Monitor market conditions.
        """)
        
    elif result['predicted_class'] == 'N':
        st.warning("""
        **⚖️ CAUTIOUS APPROACH RECOMMENDATION**
        
        **Considerations:**
        - Moderate fundamentals
        - Acceptable risk levels
        - Requires careful analysis
        
        **Action:** Consider applying with moderate allocation. Review company details.
        """)
        
    else:
        st.error("""
        **🚨 HIGH RISK - AVOID RECOMMENDATION**
        
        **Concerns Identified:**
        - Weak fundamental metrics
        - High risk factors
        - Poor growth indicators
        
        **Action:** Avoid application. Wait for better opportunities.
        """)
    
    # GMP-specific advice
    if gmp_ratio > 50:
        st.info(f"💡 **GMP Insight:** High GMP ({gmp_ratio}) indicates strong market demand. This has been factored into the positive adjustment.")

        
def show_model_analysis():
    st.header("📊 Model Performance Analysis")
    
    # Load the pre-trained model artifacts
    try:
        artifacts = load_all_models()
    except Exception:
        st.error("❌ Models not found. Please load the models first in 'Model Training' tab.")
        return
    
    # Try to load performance data and test results
    try:
        # Assuming model_performance.pkl is also in the same directory, but script had it absolute?
        # Checked earlier, it wasn't loaded in load_all_models, but used here.
        # Need to fix this path too!
        with open('model_performance.pkl', 'rb') as f:
            individual_performance = pickle.load(f)
        
        # Try to load the original data for test set recreation
        df = load_and_preprocess_data()
        if df is not None:
            X, y_encoded, _, original_features = advanced_feature_engineering(df)
            # Recreate the test set to match the training
            _, X_test, _, y_test_encoded = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Convert encoded test labels back to original
            y_test_original = artifacts['label_encoder'].inverse_transform(y_test_encoded)
            
            # Display full performance analysis with visualizations
            display_model_performance(individual_performance, y_test_original)
        else:
            st.warning("⚠️ Could not load training data for detailed analysis")
            display_basic_model_info(artifacts, individual_performance)
            
    except FileNotFoundError:
        st.warning("⚠️ Performance metrics file not found. Showing basic model information.")
        display_basic_model_info(artifacts, None)


def display_basic_model_info(artifacts, individual_performance=None):
    """Display basic model information when full performance data is unavailable"""
    
    st.subheader("🔍 Model Architecture Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Components:**")
        st.write("- Primary Model: Advanced Ensemble")
        st.write("- Preprocessing: PowerTransformer")
        st.write("- Feature Selection: RFECV")
        st.write(f"- Selected Features: {len(artifacts['selected_features'])}")
        st.write(f"- Classes: {', '.join(artifacts['label_encoder'].classes_)}")
    
    with col2:
        st.write("**Feature Engineering:**")
        st.write("- Interaction Features")
        st.write("- Profitability Metrics")
        st.write("- Growth Momentum Indicators")
        st.write("- Risk & Stability Measures")
        st.write("- Non-linear Transformations")
    
    # Display selected features
    st.subheader("📋 Selected Features")
    
    features_df = pd.DataFrame({
        'Feature Name': artifacts['selected_features'],
        'Index': range(len(artifacts['selected_features']))
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(features_df, height=400)
    
    with col2:
        st.write("**Feature Categories:**")
        
        # Categorize features
        base_features = [f for f in artifacts['selected_features'] if any(
            key in f for key in ['P/E', 'Market', 'Dividend', 'Profit', 'Sales', 'ROCE', 'Issue']
        )]
        engineered_features = [f for f in artifacts['selected_features'] if f not in base_features]
        
        st.write(f"- Base Features: {len(base_features)}")
        st.write(f"- Engineered Features: {len(engineered_features)}")
        st.write(f"- **Total: {len(artifacts['selected_features'])}**")
    
    # If we have basic performance data, display it
    if individual_performance:
        st.subheader("📈 Model Performance Summary")
        display_performance_summary(individual_performance)
    
    # Feature importance visualization (if available)
    if hasattr(artifacts['model'], 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        display_feature_importance(artifacts['model'], artifacts['selected_features'])


def display_model_performance(individual_performance, y_test_original):
    """Display comprehensive model performance summary with all visualizations"""
    
    st.header("📊 Comprehensive Model Performance Analysis")
    
    # Create performance dataframe with safe access to metrics
    performance_data = []
    for model_name, metrics in individual_performance.items():
        # Safely get metrics with default values
        model_entry = {
            'Model': model_name.upper() if model_name not in ['stacking', 'voting'] else f"{model_name.title()} Ensemble",
            'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.3f}"
        }
        
        # Add optional metrics if available
        if 'precision' in metrics:
            model_entry['Precision'] = f"{metrics['precision']:.3f}"
        if 'recall' in metrics:
            model_entry['Recall'] = f"{metrics['recall']:.3f}"
        
        performance_data.append(model_entry)
    
    # Display performance table
    st.subheader("🎯 Model Accuracy Comparison")
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, width='stretch')
    
    # Visual comparison
    st.subheader("📈 Model Performance Visualization")
    
    # Separate base models and ensemble models
    base_models = [k for k in individual_performance.keys() if k not in ['stacking', 'voting']]
    ensemble_models = [k for k in individual_performance.keys() if k in ['stacking', 'voting']]
    
    # Check if we have models to display
    if not base_models and not ensemble_models:
        st.warning("⚠️ No model performance data available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison for base models
    if base_models:
        base_accuracies = [individual_performance[model].get('accuracy', 0) for model in base_models]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(base_models)]
        
        axes[0, 0].bar(base_models, base_accuracies, color=colors)
        axes[0, 0].set_title('Base Models Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
    else:
        axes[0, 0].text(0.5, 0.5, 'No base models available', ha='center', va='center')
        axes[0, 0].axis('off')
    
    # Ensemble comparison
    if ensemble_models:
        ensemble_accuracies = [individual_performance[model].get('accuracy', 0) for model in ensemble_models]
        
        axes[0, 1].bar(ensemble_models, ensemble_accuracies, color=['#A8E6CF', '#DCEDC1'])
        axes[0, 1].set_title('Ensemble Models Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, 'No ensemble models available', ha='center', va='center')
        axes[0, 1].axis('off')
    
    # F1-Score comparison
    if base_models:
        base_f1 = [individual_performance[model].get('f1_score', 0) for model in base_models]
        axes[1, 0].bar(base_models, base_f1, color=colors)
        axes[1, 0].set_title('Base Models F1-Score', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    else:
        axes[1, 0].text(0.5, 0.5, 'No base models available', ha='center', va='center')
        axes[1, 0].axis('off')
    
    # Best model highlight
    best_model = max(individual_performance.items(), key=lambda x: x[1].get('accuracy', 0))
    best_accuracy = best_model[1].get('accuracy', 0)
    best_f1 = best_model[1].get('f1_score', 0)
    
    axes[1, 1].text(0.1, 0.5, 
                   f"🏆 Best Model: {best_model[0].upper()}\n\nAccuracy: {best_accuracy:.3f}\nF1-Score: {best_f1:.3f}", 
                   fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="darkblue", linewidth=2))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Detailed metrics for best model
    st.subheader("🥇 Best Performing Model Details")
    best_model_name = best_model[0]
    best_metrics = best_model[1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.3f}")
    with col2:
        st.metric("F1-Score", f"{best_metrics.get('f1_score', 0):.3f}")
    with col3:
        st.metric("Precision", f"{best_metrics.get('precision', 0):.3f}")
    with col4:
        st.metric("Recall", f"{best_metrics.get('recall', 0):.3f}")
    
    # Confusion Matrix for best model
    st.subheader("📋 Confusion Matrix - Best Model")
    if 'predictions' in best_metrics:
        cm = confusion_matrix(y_test_original, best_metrics['predictions'], labels=['S', 'F', 'N'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Success', 'Fail', 'Normal'],
                   yticklabels=['Success', 'Fail', 'Normal'],
                   cbar_kws={'label': 'Count'},
                   linewidths=1, linecolor='gray')
        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.ylabel('Actual', fontsize=12, fontweight='bold')
        plt.title(f'Confusion Matrix - {best_model_name.upper()}', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    # All models comparison heatmap
    st.subheader("🔥 All Models Performance Heatmap")
    display_performance_heatmap(individual_performance)
    
    # Classification report
    st.subheader("📊 Detailed Classification Report - Best Model")
    if 'predictions' in best_metrics:
        from sklearn.metrics import classification_report
        report = classification_report(y_test_original, best_metrics['predictions'], 
                                      labels=['S', 'F', 'N'],
                                      target_names=['Success', 'Fail', 'Normal'])
        st.text(report)


def display_performance_summary(individual_performance):
    """Display basic performance summary without full visualizations"""
    
    # Create simple performance table
    performance_data = []
    for model_name, metrics in individual_performance.items():
        performance_data.append({
            'Model': model_name.upper(),
            'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.3f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, width='stretch')


def display_performance_heatmap(individual_performance):
    """Display a heatmap comparing all models across all metrics"""
    
    # Prepare data for heatmap with safe metric access
    models = []
    metrics_data = []
    available_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    
    # First pass: determine which metrics are available across all models
    metrics_present = {metric: False for metric in available_metrics}
    for metrics in individual_performance.values():
        for metric in available_metrics:
            if metric in metrics and metrics[metric] > 0:  # Only count if value exists and > 0
                metrics_present[metric] = True
    
    # Build column names for available metrics
    column_names = []
    metric_keys = []
    if metrics_present['accuracy']:
        column_names.append('Accuracy')
        metric_keys.append('accuracy')
    if metrics_present['f1_score']:
        column_names.append('F1-Score')
        metric_keys.append('f1_score')
    if metrics_present['precision']:
        column_names.append('Precision')
        metric_keys.append('precision')
    if metrics_present['recall']:
        column_names.append('Recall')
        metric_keys.append('recall')
    
    # If no metrics available, show warning
    if not column_names:
        st.warning("⚠️ No performance metrics available for heatmap visualization")
        return
    
    # If precision/recall are missing, show info
    if not metrics_present['precision'] or not metrics_present['recall']:
        st.info("ℹ️ **Note:** Precision and Recall metrics are not available for all models. Re-train the model to generate complete metrics.")
    
    # Second pass: collect data (only for models that have the metrics)
    for model_name, metrics in individual_performance.items():
        # Check if this model has at least accuracy
        if 'accuracy' not in metrics or metrics['accuracy'] == 0:
            continue
            
        models.append(model_name.upper())
        row_data = []
        
        for key in metric_keys:
            row_data.append(metrics.get(key, 0))
        
        metrics_data.append(row_data)
    
    if not metrics_data:
        st.warning("⚠️ No valid model data available for visualization")
        return
    
    metrics_df = pd.DataFrame(
        metrics_data,
        index=models,
        columns=column_names
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=1)
    plt.title('Model Performance Comparison Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Models', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_feature_importance(model, feature_names):
    """Display feature importance chart"""
    
    try:
        importances = model.feature_importances_
        
        # Create DataFrame
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], 
                       color=plt.cm.viridis(feature_imp_df['Importance'] / feature_imp_df['Importance'].max()))
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, feature_imp_df['Importance'])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show full table
        with st.expander("📋 View All Feature Importances"):
            full_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Add ranking
            full_importance_df['Rank'] = range(1, len(full_importance_df) + 1)
            full_importance_df = full_importance_df[['Rank', 'Feature', 'Importance']]
            
            st.dataframe(full_importance_df, width='stretch', height=400)
            
    except Exception as e:
        st.info(f"Feature importance not available for this model type: {type(model).__name__}")


# Additional helper functions for data loading (needed for analysis)
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the IPO dataset"""
    try:
        df = pd.read_csv('data.csv')
        st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Replace NA values and handle missing data
        df = df.replace(['NA', '', 'NaN', 'null'], 0)
        df = df.fillna(0)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None


def advanced_feature_engineering(df):
    """Advanced feature engineering with domain knowledge"""
    
    # Define core feature columns
    feature_columns = [
        'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
        'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
        'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
        'Issue Price (Rs)', 'ROCE %'
    ]
    
    # Extract features and target
    X = df[feature_columns].copy()
    y = df['Classification'].copy()
    
    # Convert features to numeric
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Convert target and filter valid classes
    y = y.astype(str).str.strip()
    valid_classes = ['S', 'F', 'N']
    mask = y.isin(valid_classes)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    # Encode labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Apply feature engineering (same as in prediction)
    X = apply_feature_engineering(X)
    
    return X, y_encoded, label_encoder, feature_columns

if __name__ == "__main__":
    main()
