import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1500

data = []

for i in range(rows):

    # -------------------------
    # FEATURES
    # -------------------------

    credit_score = np.random.randint(1, 11)

    income = np.random.randint(10000, 120000)

    employment = np.random.choice([1, 2, 3])  
    # 1=Salaried, 2=Self-employed, 3=Unemployed

    loan_amount = np.random.randint(50000, 2000000)

    previous_default = np.random.choice([0, 1])

    num_loans = np.random.randint(0, 10)

    loan_duration = np.random.randint(1, 11)

    credit_cards = np.random.randint(0, 15)

    late_payments = np.random.randint(0, 6)

    total_assets = np.random.randint(50000, 5000000)

    dti = np.random.randint(10, 100)

    # -------------------------
    # RISK SCORE LOGIC
    # -------------------------

    risk = 0

    # BAD CONDITIONS
    if credit_score <= 3:
        risk += 2

    if income < 30000:
        risk += 2

    if employment == 3:
        risk += 2

    if previous_default == 1:
        risk += 3

    if loan_amount > income * 15:
        risk += 2

    if num_loans > 5:
        risk += 1

    if late_payments >= 3:
        risk += 2

    if dti > 60:
        risk += 2

    if total_assets < 100000:
        risk += 1

    # GOOD CONDITIONS
    if credit_score >= 8:
        risk -= 2

    if income > 70000:
        risk -= 2

    if employment == 1:
        risk -= 1

    if previous_default == 0:
        risk -= 1

    if dti < 30:
        risk -= 1

    # -------------------------
    # LABEL CREATION
    # -------------------------

    # 1 = HIGH RISK
    # 0 = LOW RISK

    if risk >= 5:
        label = 1
    else:
        label = 0

    data.append([
        label,
        i + 1,
        credit_score,
        income,
        employment,
        loan_amount,
        previous_default,
        num_loans,
        loan_duration,
        credit_cards,
        late_payments,
        total_assets,
        dti
    ])

# -------------------------
# DATAFRAME
# -------------------------

columns = [
    "label",
    "id",
    "fea_1",
    "fea_2",
    "fea_3",
    "fea_4",
    "fea_5",
    "fea_6",
    "fea_7",
    "fea_8",
    "fea_9",
    "fea_10",
    "fea_11"
]

df = pd.DataFrame(data, columns=columns)

# -------------------------
# SAVE
# -------------------------

df.to_excel("better_credit_dataset.xlsx", index=False)

print("Dataset Generated Successfully!")