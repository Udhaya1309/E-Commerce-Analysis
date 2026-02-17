"""
AI Marketing Agent - ML-driven customer segmentation and personalized email generation.
Uses model predictions (not rule-based logic) for strategy decisions.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Local config
try:
    from config import (
        DATA_PATH,
        OPENAI_API_KEY,
        OPENAI_MODEL,
        RANDOM_STATE,
        TEST_SIZE,
        HIGH_PROB_THRESHOLD,
        LOW_PROB_THRESHOLD,
    )
except ImportError:
    DATA_PATH = Path("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    OPENAI_API_KEY = ""
    OPENAI_MODEL = "gpt-4o-mini"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    HIGH_PROB_THRESHOLD = 0.6
    LOW_PROB_THRESHOLD = 0.35

# Columns used for ML (must match training)
NUMERIC_FEATURES = [
    "Age",
    "Frequency_of_Purchase",
    "Brand_Loyalty",
    "Product_Rating",
    "Time_Spent_on_Product_Research(hours)",
    "Return_Rate",
    "Customer_Satisfaction",
    "Time_to_Decision",
]
# Purchase_Amount added after cleaning
CATEGORICAL_FEATURES = [
    "Gender",
    "Income_Level",
    "Engagement_with_Ads",
    "Purchase_Category",
    "Purchase_Channel",
    "Discount_Sensitivity",
    "Customer_Loyalty_Program_Member",  # bool-like, will encode
    "Discount_Used",
]
TARGET_COLUMN = "Purchase_Intent"


def _clean_purchase_amount(series: pd.Series) -> pd.Series:
    """Remove currency symbol and whitespace, convert to float."""
    def clean(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().replace("$", "").replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.map(clean)


def _encode_bool_like(series: pd.Series) -> pd.Series:
    """Map TRUE/FALSE and similar to 1/0."""
    return series.astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)


class MarketingAIAgent:
    """
    Intelligent marketing agent: Data → ML prediction → Strategy decision → Email generation.
    All decisions are driven by model outputs and probabilities, not fixed rules.
    """

    def __init__(self, data_path: Optional[Path] = None, use_openai: bool = True):
        self.data_path = data_path or DATA_PATH
        self.use_openai = bool(use_openai and OPENAI_API_KEY)
        self._raw_df: Optional[pd.DataFrame] = None
        self._df: Optional[pd.DataFrame] = None
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self._preprocessor: Optional[ColumnTransformer] = None
        self._model: Optional[GradientBoostingClassifier] = None
        self._feature_names_out: Optional[List[str]] = None
        self._accuracy: Optional[float] = None
        self._class_names: Optional[List[str]] = None

    def preprocess_data(self) -> pd.DataFrame:
        """
        Load CSV, clean purchase amount, handle missing values,
        encode categoricals, and prepare feature matrix.
        """
        self._raw_df = pd.read_csv(self.data_path)
        df = self._raw_df.copy()

        # Clean Purchase_Amount: remove $ and convert to float
        if "Purchase_Amount" in df.columns:
            df["Purchase_Amount"] = _clean_purchase_amount(df["Purchase_Amount"])

        # Add cleaned numeric column for modeling
        NUMERIC_FEATURES_WITH_AMOUNT = NUMERIC_FEATURES + ["Purchase_Amount"]

        # Encode bool-like columns for ML
        for col in ["Customer_Loyalty_Program_Member", "Discount_Used"]:
            if col in df.columns:
                df[col] = _encode_bool_like(df[col])

        # Drop rows where target is missing
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not in dataset")
        df = df.dropna(subset=[TARGET_COLUMN])

        # Fill numeric missing with median
        for c in NUMERIC_FEATURES_WITH_AMOUNT:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df[c] = df[c].fillna(df[c].median())

        # Fill categorical missing with mode
        for c in CATEGORICAL_FEATURES:
            if c in df.columns and df[c].dtype == object:
                df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown")
            elif c in df.columns:
                df[c] = df[c].fillna(0)

        self._df = df
        self._y = df[TARGET_COLUMN]
        self._class_names = sorted(self._y.unique().tolist())

        # Build feature subset for model
        feature_cols = [c for c in (NUMERIC_FEATURES_WITH_AMOUNT + CATEGORICAL_FEATURES) if c in df.columns]
        self._X = df[feature_cols].copy()

        # Store preprocessor for transform at predict time
        num_cols = [c for c in NUMERIC_FEATURES_WITH_AMOUNT if c in self._X.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in self._X.columns]

        self._preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_transformed = self._preprocessor.fit_transform(self._X)
        self._feature_names_out = self._preprocessor.get_feature_names_out().tolist()
        self._X = pd.DataFrame(X_transformed, columns=self._feature_names_out, index=self._df.index)

        return self._df

    def train_model(self) -> float:
        """
        Train GradientBoostingClassifier on preprocessed data.
        Returns test accuracy.
        """
        if self._X is None or self._y is None:
            self.preprocess_data()

        X_train, X_test, y_train, y_test = train_test_split(
            self._X, self._y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=self._y
        )
        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE,
            min_samples_leaf=5,
        )
        self._model.fit(X_train, y_train)
        y_pred = self._model.predict(X_test)
        self._accuracy = float(accuracy_score(y_test, y_pred))
        print("Model trained.")
        print(f"Test accuracy: {self._accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=self._class_names))
        return self._accuracy

    def _customer_data_to_features(self, customer_data: Dict[str, Any]) -> np.ndarray:
        """Convert a single customer dict into the same feature space as training."""
        if self._preprocessor is None or self._model is None:
            raise RuntimeError("Call preprocess_data() and train_model() first.")

        # Build one row with same columns as training
        row = {}
        for c in NUMERIC_FEATURES + ["Purchase_Amount"]:
            row[c] = customer_data.get(c, np.nan)
        for c in CATEGORICAL_FEATURES:
            if c in ("Customer_Loyalty_Program_Member", "Discount_Used"):
                row[c] = 1 if customer_data.get(c) in (True, "TRUE", "True", "1", 1) else 0
            else:
                row[c] = customer_data.get(c, "Unknown")
        df_one = pd.DataFrame([row])
        # Ensure numeric types
        for c in NUMERIC_FEATURES + ["Purchase_Amount"]:
            if c in df_one.columns:
                df_one[c] = pd.to_numeric(df_one[c], errors="coerce")
        X = self._preprocessor.transform(df_one)
        return pd.DataFrame(X, columns=self._feature_names_out)

    def predict_customer_segment(self, customer_data: Dict[str, Any]) -> Tuple[str, float, np.ndarray]:
        """
        Predict purchase intent (segment) and confidence from customer_data.
        Returns (predicted_label, max_probability, full_probabilities).
        """
        if self._model is None:
            raise RuntimeError("Train the model first with train_model().")
        X = self._customer_data_to_features(customer_data)  # DataFrame with feature names
        pred = self._model.predict(X)[0]
        probs = self._model.predict_proba(X)[0]
        idx = list(self._model.classes_).index(pred)
        prob = float(probs[idx])
        return pred, prob, probs

    def _strategy_from_probability(self, max_prob: float) -> str:
        """Decide strategy from prediction probability (no hardcoded customer rules)."""
        if max_prob >= HIGH_PROB_THRESHOLD:
            return "Upsell"
        if max_prob <= LOW_PROB_THRESHOLD:
            return "Re-engagement"
        return "Nurture"

    def generate_email(
        self,
        customer_data: Dict[str, Any],
        prediction: str,
        strategy: str,
        probability: float,
    ) -> str:
        """
        Generate personalized marketing email using LLM.
        Uses customer name, predicted behavior, category, engagement, loyalty.
        """
        name = customer_data.get("Customer_Name", customer_data.get("customer_name", "Valued Customer"))
        category = customer_data.get("Purchase_Category", customer_data.get("purchase_category", "our products"))
        engagement = customer_data.get("Engagement_with_Ads", customer_data.get("engagement", "Unknown"))
        loyalty = customer_data.get("Customer_Loyalty_Program_Member", False)
        loyalty_status = "Loyalty member" if loyalty in (True, "TRUE", 1, "1") else "Not yet a loyalty member"

        if self.use_openai and OPENAI_API_KEY:
            return self._generate_email_openai(
                name=name,
                prediction=prediction,
                strategy=strategy,
                probability=probability,
                category=category,
                engagement=engagement,
                loyalty_status=loyalty_status,
                extra_context=customer_data,
            )
        return self._generate_email_fallback(
            name=name,
            prediction=prediction,
            strategy=strategy,
            category=category,
            engagement=engagement,
            loyalty_status=loyalty_status,
        )

    def _generate_email_openai(
        self,
        name: str,
        prediction: str,
        strategy: str,
        probability: float,
        category: str,
        engagement: str,
        loyalty_status: str,
        extra_context: Dict,
    ) -> str:
        """Call OpenAI API to generate natural, non-templated email."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"""You are a skilled marketing copywriter. Write ONE short, personalized marketing email (2-4 paragraphs) for this customer.

Customer name: {name}
Predicted purchase behavior (from our ML model): {prediction}
Confidence of prediction: {probability:.0%}
Recommended strategy: {strategy}
Purchase category of interest: {category}
Engagement with ads: {engagement}
Loyalty status: {loyalty_status}

Strategy meaning:
- Upsell: they are likely to buy; suggest premium or complementary products.
- Nurture: build relationship with helpful content and gentle offers.
- Re-engagement: win them back with a compelling offer or reminder of value.

Write in a natural, human tone. Do NOT use bullet points or placeholders like [NAME]. Start with a greeting and end with a sign-off. No subject line. Output only the email body."""
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            return self._generate_email_fallback(
                name=name,
                prediction=prediction,
                strategy=strategy,
                category=category,
                engagement=engagement,
                loyalty_status=loyalty_status,
            ) + f"\n\n[Email generation note: API error - {e}]"

    def _generate_email_fallback(
        self,
        name: str,
        prediction: str,
        strategy: str,
        category: str,
        engagement: str,
        loyalty_status: str,
    ) -> str:
        """Intelligent fallback when API is unavailable: varied, context-aware text (not a fixed template)."""
        strategy_lines = {
            "Upsell": (
                f"Given your interest in {category} and your engagement level ({engagement}), "
                "we thought you might love our premium range or a bundle that fits your style."
            ),
            "Nurture": (
                f"We see you've been exploring {category}. Here’s a quick guide we put together, "
                "plus a small thank-you offer—no pressure, just something we think you’ll find useful."
            ),
            "Re-engagement": (
                "We’ve missed you. We’ve updated our selection and have a special offer on "
                f"{category} we thought might bring you back. We’d love to hear from you."
            ),
        }
        line = strategy_lines.get(strategy, strategy_lines["Nurture"])
        return (
            f"Hi {name},\n\n"
            f"Our systems suggest your purchase style is best described as {prediction}. {line}\n\n"
            f"As a {loyalty_status}, you’re always on our radar. Reply to this email if you’d like to adjust your preferences.\n\n"
            "Best regards,\nThe Marketing Team"
        )

    def run_agent(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full pipeline: Input → Predict → Decide strategy → Generate email → Output.
        """
        prediction, probability, _ = self.predict_customer_segment(customer_data)
        strategy = self._strategy_from_probability(probability)
        email = self.generate_email(customer_data, prediction, strategy, probability)
        return {
            "prediction": prediction,
            "probability": probability,
            "strategy": strategy,
            "email": email,
        }


def get_sample_customers() -> List[Dict[str, Any]]:
    """Return 3 sample customers: Loyal, At-risk, New (for demo)."""
    return [
        {
            "Customer_Name": "Sarah Chen",
            "Age": 34,
            "Gender": "Female",
            "Income_Level": "High",
            "Engagement_with_Ads": "High",
            "Frequency_of_Purchase": 10,
            "Purchase_Amount": 420.0,
            "Customer_Loyalty_Program_Member": True,
            "Discount_Used": True,
            "Time_to_Decision": 2,
            "Purchase_Category": "Electronics",
            "Purchase_Channel": "Online",
            "Brand_Loyalty": 5,
            "Product_Rating": 5,
            "Time_Spent_on_Product_Research(hours)": 1.5,
            "Return_Rate": 0,
            "Customer_Satisfaction": 9,
            "Discount_Sensitivity": "Not Sensitive",
        },
        {
            "Customer_Name": "James Miller",
            "Age": 45,
            "Gender": "Male",
            "Income_Level": "Middle",
            "Engagement_with_Ads": "None",
            "Frequency_of_Purchase": 2,
            "Purchase_Amount": 89.0,
            "Customer_Loyalty_Program_Member": False,
            "Discount_Used": False,
            "Time_to_Decision": 14,
            "Purchase_Category": "Office Supplies",
            "Purchase_Channel": "In-Store",
            "Brand_Loyalty": 2,
            "Product_Rating": 3,
            "Time_Spent_on_Product_Research(hours)": 0.5,
            "Return_Rate": 2,
            "Customer_Satisfaction": 4,
            "Discount_Sensitivity": "Very Sensitive",
        },
        {
            "Customer_Name": "Alex Rivera",
            "Age": 26,
            "Gender": "Male",
            "Income_Level": "Middle",
            "Engagement_with_Ads": "Medium",
            "Frequency_of_Purchase": 4,
            "Purchase_Amount": 180.0,
            "Customer_Loyalty_Program_Member": False,
            "Discount_Used": True,
            "Time_to_Decision": 8,
            "Purchase_Category": "Clothing",
            "Purchase_Channel": "Mixed",
            "Brand_Loyalty": 3,
            "Product_Rating": 4,
            "Time_Spent_on_Product_Research(hours)": 1.0,
            "Return_Rate": 1,
            "Customer_Satisfaction": 7,
            "Discount_Sensitivity": "Somewhat Sensitive",
        },
    ]
