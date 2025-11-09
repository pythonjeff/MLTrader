# data/test_cb.py
from cb_data import get_cb_client
import os

def list_perpetual_products(limit=100, offset=0):
    client = get_cb_client()
    params = {
        "limit": str(limit),
        "offset": str(offset),
        "product_type": "FUTURE",
        "contract_expiry_type": "PERPETUAL"
    }
    response = client.get("/api/v3/brokerage/products", params=params)
    data = response if isinstance(response, dict) else response.to_dict()
    products = data.get("products", [])
    if not products:
        print("No perpetual products found.")
    for p in products:
        print(p.get("product_id"), "–", p.get("display_name"))

if __name__ == "__main__":
    list_perpetual_products()