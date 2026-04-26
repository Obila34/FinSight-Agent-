from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path

fake = Faker()
random.seed(42)

project_root = Path(__file__).resolve().parent.parent
transactions_path = project_root / "data" / "transactions.csv"

# Load your real Plaid transactions first
plaid_df = pd.read_csv(transactions_path)
print(f"Plaid transactions: {len(plaid_df)}")

# Generate synthetic Nairobi-flavoured transactions
merchants = {
    "Food": ["Naivas Supermarket", "Carrefour", "Java House", "KFC Nairobi", "Artcaffe", "Mama Oliech", "Subway Westgate", "Chicken Inn"],
    "Transport": ["Uber", "Bolt", "Little Cab", "Matatu Fare", "Shell Petrol", "Total Energies", "Kenya Bus"],
    "Entertainment": ["Netflix", "Showmax", "IMAX Westgate", "Spotify", "DStv", "Steam"],
    "Health": ["Goodlife Pharmacy", "AAR Clinic", "Nairobi Hospital", "Haltons Pharmacy"],
    "Shopping": ["Jumia", "Kilimall", "Mr Price", "Zara Village Market", "Woolworths", "Nike Town"],
    "Utilities": ["Kenya Power", "Safaricom", "Zuku Fibre", "Nairobi Water", "Airtel Kenya"],
    "Rent": ["Rent Payment"],
    "Other": ["ATM Withdrawal", "Bank Transfer", "M-Pesa Send Money"]
}

records = []
start_date = datetime.today() - timedelta(days=180)

for i in range(584):  # 584 + 16 Plaid = 600 total
    category = random.choices(
        list(merchants.keys()),
        weights=[30, 20, 10, 8, 12, 10, 5, 5]
    )[0]
    merchant = random.choice(merchants[category])
    
    # Realistic Nairobi amounts in KES
    amount_ranges = {
        "Food": (150, 2500),
        "Transport": (50, 800),
        "Entertainment": (300, 1500),
        "Health": (200, 8000),
        "Shopping": (500, 12000),
        "Utilities": (500, 5000),
        "Rent": (15000, 45000),
        "Other": (500, 10000)
    }
    
    low, high = amount_ranges[category]
    amount = round(random.uniform(low, high), 2)
    date = start_date + timedelta(days=random.randint(0, 180))
    
    records.append({
        "date": date.strftime("%Y-%m-%d"),
        "merchant": merchant,
        "amount": amount,
        "category": category
    })

synthetic_df = pd.DataFrame(records)

# Combine both
combined_df = pd.concat([plaid_df, synthetic_df], ignore_index=True)
combined_df = combined_df.sort_values("date").reset_index(drop=True)
combined_df.to_csv(transactions_path, index=False)

print(f"Synthetic transactions generated: {len(synthetic_df)}")
print(f"Total combined transactions: {len(combined_df)}")
print(combined_df.head(10))