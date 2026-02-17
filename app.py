"""
Streamlit UI for the AI Marketing Agent.
Input customer details, click "Run AI Agent", see prediction, strategy, and generated email.
"""

import streamlit as st
from marketing_agent import MarketingAIAgent, get_sample_customers

# Page config
st.set_page_config(page_title="AI Marketing Agent", page_icon="📧", layout="wide")
st.title("📧 AI Marketing Agent")
st.caption("ML-driven customer prediction and personalized email generation")

# Load or train agent once
@st.cache_resource
def get_agent():
    agent = MarketingAIAgent()
    agent.preprocess_data()
    agent.train_model()
    return agent

with st.spinner("Loading data and training model..."):
    agent = get_agent()

st.success("Model ready. Enter customer details below or load a sample.")

# Sidebar: choose input mode
input_mode = st.sidebar.radio("Input", ["Manual", "Sample customers"])
sample_names = ["Loyal (Sarah)", "At-risk (James)", "New (Alex)"]
samples = get_sample_customers()

if input_mode == "Sample customers":
    idx = st.sidebar.selectbox("Pick sample", range(3), format_func=lambda i: sample_names[i])
    customer = samples[idx]
    # Show as form for editing
    with st.expander("Edit sample customer data", expanded=False):
        customer["Customer_Name"] = st.text_input("Name", value=customer.get("Customer_Name", ""))
        customer["Age"] = st.number_input("Age", min_value=18, max_value=100, value=customer.get("Age", 30))
        customer["Income_Level"] = st.selectbox("Income", ["Low", "Middle", "High"], index=["Low", "Middle", "High"].index(customer.get("Income_Level", "Middle")))
        customer["Engagement_with_Ads"] = st.selectbox("Engagement with ads", ["None", "Low", "Medium", "High"], index=["None", "Low", "Medium", "High"].index(customer.get("Engagement_with_Ads", "Medium")))
        customer["Frequency_of_Purchase"] = st.number_input("Purchase frequency (per period)", min_value=0, value=customer.get("Frequency_of_Purchase", 5))
        customer["Purchase_Amount"] = st.number_input("Purchase amount", min_value=0.0, value=float(customer.get("Purchase_Amount", 200)))
        customer["Customer_Loyalty_Program_Member"] = st.checkbox("Loyalty member", value=customer.get("Customer_Loyalty_Program_Member", False))
        customer["Discount_Used"] = st.checkbox("Uses discounts", value=customer.get("Discount_Used", False))
        customer["Time_to_Decision"] = st.number_input("Time to decision (e.g. days)", min_value=0, value=customer.get("Time_to_Decision", 5))
        customer["Purchase_Category"] = st.text_input("Purchase category", value=customer.get("Purchase_Category", "Electronics"))
        customer["Brand_Loyalty"] = st.slider("Brand loyalty (1-5)", 1, 5, customer.get("Brand_Loyalty", 3))
        customer["Product_Rating"] = st.slider("Product rating (1-5)", 1, 5, customer.get("Product_Rating", 4))
        customer["Time_Spent_on_Product_Research(hours)"] = st.number_input("Research hours", min_value=0.0, value=float(customer.get("Time_Spent_on_Product_Research(hours)", 1.0)))
        customer["Return_Rate"] = st.number_input("Return rate", min_value=0, value=customer.get("Return_Rate", 0))
        customer["Customer_Satisfaction"] = st.slider("Satisfaction (1-10)", 1, 10, customer.get("Customer_Satisfaction", 7))
        customer["Discount_Sensitivity"] = st.selectbox("Discount sensitivity", ["Not Sensitive", "Somewhat Sensitive", "Very Sensitive"], index=1)
else:
    st.sidebar.markdown("Fill the form and click **Run AI Agent**.")
    customer = {
        "Customer_Name": st.text_input("Customer name", value="Valued Customer"),
        "Age": st.number_input("Age", min_value=18, max_value=100, value=35),
        "Income_Level": st.selectbox("Income level", ["Low", "Middle", "High"]),
        "Engagement_with_Ads": st.selectbox("Engagement with ads", ["None", "Low", "Medium", "High"]),
        "Frequency_of_Purchase": st.number_input("Purchase frequency", min_value=0, value=5),
        "Purchase_Amount": st.number_input("Purchase amount", min_value=0.0, value=250.0),
        "Customer_Loyalty_Program_Member": st.checkbox("Loyalty program member", value=False),
        "Discount_Used": st.checkbox("Uses discounts", value=True),
        "Time_to_Decision": st.number_input("Time to decision", min_value=0, value=5),
        "Purchase_Category": st.text_input("Purchase category", value="Electronics"),
        "Brand_Loyalty": st.slider("Brand loyalty (1-5)", 1, 5, 3),
        "Product_Rating": st.slider("Product rating (1-5)", 1, 5, 4),
        "Time_Spent_on_Product_Research(hours)": st.number_input("Research hours", min_value=0.0, value=1.0),
        "Return_Rate": st.number_input("Return rate", min_value=0, value=0),
        "Customer_Satisfaction": st.slider("Satisfaction (1-10)", 1, 10, 7),
        "Discount_Sensitivity": st.selectbox("Discount sensitivity", ["Not Sensitive", "Somewhat Sensitive", "Very Sensitive"]),
    }

if st.button("Run AI Agent", type="primary"):
    with st.spinner("Predicting and generating email..."):
        try:
            result = agent.run_agent(customer)
            st.subheader("Prediction")
            st.metric("Predicted intent", result["prediction"])
            st.metric("Confidence", f"{result['probability']:.0%}")
            st.metric("Strategy", result["strategy"])
            st.subheader("Generated email")
            st.text_area("Email body", value=result["email"], height=280, disabled=True)
        except Exception as e:
            st.error(f"Error: {e}")
            raise
