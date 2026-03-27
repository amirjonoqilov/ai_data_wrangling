import pandas as pd
import numpy as np

np.random.seed(42)
n = 1200

categories = ["Electronics", "Clothing", "Food", "Books", "Sports", "Home"]
regions = ["North", "South", "East", "West", "Central"]
reps = [f"Rep_{i:03d}" for i in range(1, 31)]

dates = pd.date_range("2022-01-01", "2024-12-31", periods=n)

df1 = pd.DataFrame({
    "order_id": range(10001, 10001 + n),
    "date": dates.strftime("%Y-%m-%d"),
    "category": np.random.choice(categories, n),
    "region": np.random.choice(regions, n),
    "sales_rep": np.random.choice(reps, n),
    "quantity": np.random.randint(1, 50, n),
    "unit_price": np.round(np.random.uniform(5, 500, n), 2),
    "discount_pct": np.random.choice([0, 5, 10, 15, 20, np.nan], n),
    "customer_rating": np.random.choice([1, 2, 3, 4, 5, np.nan], n),
    "returned": np.random.choice(["Yes", "No", "no", "YES", None], n),
})
df1["revenue"] = np.round(df1["quantity"] * df1["unit_price"] * (1 - df1["discount_pct"].fillna(0) / 100), 2)
mask = np.random.rand(n) < 0.07
df1.loc[mask, "revenue"] = np.nan
mask2 = np.random.rand(n) < 0.05
df1.loc[mask2, "unit_price"] = np.nan
df1 = pd.concat([df1, df1.sample(30)], ignore_index=True)
df1["revenue"] = df1["revenue"].apply(lambda x: f"${x:.2f}" if pd.notna(x) and np.random.rand() < 0.1 else x)

df1.to_csv("sample_data/retail_sales.csv", index=False)
print("retail_sales.csv created:", df1.shape)

n2 = 1500
departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations", "Legal"]
levels = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director"]
edu = ["Bachelor", "Master", "PhD", "Associate", "High School"]

df2 = pd.DataFrame({
    "employee_id": [f"EMP{i:05d}" for i in range(1, n2 + 1)],
    "department": np.random.choice(departments, n2),
    "level": np.random.choice(levels, n2),
    "years_experience": np.random.randint(0, 30, n2),
    "age": np.random.randint(22, 65, n2),
    "education": np.random.choice(edu, n2),
    "salary": np.round(np.random.normal(75000, 25000, n2), -2),
    "performance_score": np.random.uniform(1, 5, n2).round(2),
    "remote_days_per_week": np.random.choice([0, 1, 2, 3, 4, 5, np.nan], n2),
    "city": np.random.choice(["New York", "San Francisco", "Chicago", "Austin", "Seattle", "Boston", "  new york", "san francisco"], n2),
    "hire_date": pd.date_range("2010-01-01", "2024-01-01", periods=n2).strftime("%Y-%m-%d"),
    "attrition": np.random.choice(["Yes", "No", None], n2, p=[0.15, 0.80, 0.05]),
})
df2.loc[np.random.rand(n2) < 0.06, "salary"] = np.nan
df2.loc[np.random.rand(n2) < 0.04, "age"] = np.nan
df2.loc[np.random.rand(n2) < 0.05, "performance_score"] = np.nan
outlier_idx = np.random.choice(n2, 20, replace=False)
df2.loc[outlier_idx, "salary"] = np.random.choice([5000, 500000, -1000], 20)
df2 = pd.concat([df2, df2.sample(40)], ignore_index=True)

df2.to_csv("sample_data/employee_analytics.csv", index=False)
print("employee_analytics.csv created:", df2.shape)