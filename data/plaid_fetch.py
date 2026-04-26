import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.sandbox_public_token_create_request import SandboxPublicTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
import os
from pathlib import Path
import time

from plaid.exceptions import ApiException

# Load .env from the project root regardless of current working directory.
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env", override=True)

client_id = (os.getenv("PLAID_CLIENT_ID") or "").strip()
secret = (os.getenv("PLAID_SECRET") or "").strip()
if not client_id or not secret:
    raise RuntimeError(
        "Missing Plaid credentials. Set PLAID_CLIENT_ID and PLAID_SECRET in your environment or .env file."
    )

configuration = plaid.Configuration(
    host=plaid.Environment.Sandbox,
    api_key={"clientId": client_id, "secret": secret}
)

client = plaid_api.PlaidApi(plaid.ApiClient(configuration))

# Create sandbox token
pt_request = SandboxPublicTokenCreateRequest(
    institution_id="ins_109508",
    initial_products=[Products("transactions")]
)
pt_response = client.sandbox_public_token_create(pt_request)

# Exchange for access token
exchange_request = ItemPublicTokenExchangeRequest(public_token=pt_response.public_token)
exchange_response = client.item_public_token_exchange(exchange_request)
access_token = exchange_response.access_token

# Fetch transactions
start_date = date.today() - timedelta(days=180)
end_date = date.today()
request = TransactionsGetRequest(access_token=access_token, start_date=start_date, end_date=end_date)

response = None
for attempt in range(3):
    try:
        response = client.transactions_get(request)
        break
    except ApiException as exc:
        body = getattr(exc, "body", "") or ""
        if "PRODUCT_NOT_READY" not in body or attempt == 2:
            raise
        time.sleep(2 ** attempt)

if response is None:
    raise RuntimeError("Plaid transactions were not ready after retries.")

transactions = response.transactions
records = [{"date": t.date, "merchant": t.merchant_name or t.name, "amount": t.amount, "category": t.category[0] if t.category else "Unknown"} for t in transactions]

df = pd.DataFrame(records)
output_path = project_root / "data" / "transactions.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} transactions")