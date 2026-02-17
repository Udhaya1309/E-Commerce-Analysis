"""
E-commerce Consumer Behavior Analysis - Visualization Script
Generates all analysis images for the README and project documentation.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create images directory
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
sns.set_palette("husl")

# Load and clean data
df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
df["Purchase_Amount"] = df["Purchase_Amount"].astype(str).str.replace("$", "").str.replace(",", "").str.strip().astype(float)

# 1. Purchase amount distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["Purchase_Amount"], bins=40, edgecolor="white", alpha=0.85)
ax.set_xlabel("Purchase Amount ($)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Purchase Amounts")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "01_purchase_amount_distribution.png"), dpi=120, bbox_inches="tight")
plt.close()

# 2. Purchase category counts (top 12)
fig, ax = plt.subplots(figsize=(10, 6))
cat_counts = df["Purchase_Category"].value_counts().head(12)
cat_counts.plot(kind="barh", ax=ax)
ax.set_xlabel("Number of Purchases")
ax.set_title("Top Purchase Categories")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "02_purchase_categories.png"), dpi=120, bbox_inches="tight")
plt.close()

# 3. Purchase channel distribution
fig, ax = plt.subplots(figsize=(8, 6))
df["Purchase_Channel"].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90)
ax.set_ylabel("")
ax.set_title("Purchases by Channel (Online / In-Store / Mixed)")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "03_purchase_channel.png"), dpi=120, bbox_inches="tight")
plt.close()

# 4. Device used for shopping
fig, ax = plt.subplots(figsize=(8, 5))
df["Device_Used_for_Shopping"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("husl", 4))
ax.set_xlabel("Device")
ax.set_ylabel("Count")
ax.set_title("Device Used for Shopping")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "04_device_used.png"), dpi=120, bbox_inches="tight")
plt.close()

# 5. Payment method distribution
fig, ax = plt.subplots(figsize=(9, 5))
df["Payment_Method"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("husl", 5))
ax.set_xlabel("Payment Method")
ax.set_ylabel("Count")
ax.set_title("Payment Method Preference")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "05_payment_method.png"), dpi=120, bbox_inches="tight")
plt.close()

# 6. Gender vs average purchase amount
fig, ax = plt.subplots(figsize=(8, 5))
df.groupby("Gender")["Purchase_Amount"].mean().sort_values(ascending=True).plot(kind="barh", ax=ax)
ax.set_xlabel("Average Purchase Amount ($)")
ax.set_title("Average Purchase Amount by Gender")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "06_gender_purchase.png"), dpi=120, bbox_inches="tight")
plt.close()

# 7. Income level vs purchase amount (box plot)
fig, ax = plt.subplots(figsize=(8, 5))
order = ["Low", "Middle", "High"]
df_ord = df[df["Income_Level"].isin(order)]
sns.boxplot(data=df_ord, x="Income_Level", y="Purchase_Amount", order=order, ax=ax)
ax.set_title("Purchase Amount by Income Level")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "07_income_purchase.png"), dpi=120, bbox_inches="tight")
plt.close()

# 8. Customer satisfaction distribution
fig, ax = plt.subplots(figsize=(8, 5))
df["Customer_Satisfaction"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Satisfaction Score (1-10)")
ax.set_ylabel("Count")
ax.set_title("Customer Satisfaction Distribution")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "08_satisfaction.png"), dpi=120, bbox_inches="tight")
plt.close()

# 9. Discount sensitivity
fig, ax = plt.subplots(figsize=(9, 5))
df["Discount_Sensitivity"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("husl", 4))
ax.set_xlabel("Discount Sensitivity")
ax.set_ylabel("Count")
ax.set_title("Customer Discount Sensitivity")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "09_discount_sensitivity.png"), dpi=120, bbox_inches="tight")
plt.close()

# 10. Purchase intent
fig, ax = plt.subplots(figsize=(8, 5))
df["Purchase_Intent"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("husl", 4))
ax.set_xlabel("Purchase Intent")
ax.set_ylabel("Count")
ax.set_title("Purchase Intent (Need-based, Wants-based, Impulsive, Planned)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "10_purchase_intent.png"), dpi=120, bbox_inches="tight")
plt.close()

# 11. Correlation heatmap (numeric columns)
numeric_cols = ["Age", "Purchase_Amount", "Frequency_of_Purchase", "Brand_Loyalty", "Product_Rating",
                "Time_Spent_on_Product_Research(hours)", "Return_Rate", "Customer_Satisfaction", "Time_to_Decision"]
num_df = df[numeric_cols].copy()
num_df = num_df.rename(columns={"Time_Spent_on_Product_Research(hours)": "Research_Hours"})
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
ax.set_title("Correlation Between Numeric Variables")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "11_correlation_heatmap.png"), dpi=120, bbox_inches="tight")
plt.close()

# 12. Loyalty program vs purchase amount
fig, ax = plt.subplots(figsize=(7, 5))
df.groupby("Customer_Loyalty_Program_Member")["Purchase_Amount"].mean().plot(kind="bar", ax=ax)
ax.set_xlabel("Loyalty Program Member")
ax.set_ylabel("Average Purchase Amount ($)")
ax.set_title("Average Purchase by Loyalty Program Membership")
ax.set_xticklabels(["No", "Yes"], rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "12_loyalty_purchase.png"), dpi=120, bbox_inches="tight")
plt.close()

print(f"All analysis images saved to '{IMAGES_DIR}/'")
