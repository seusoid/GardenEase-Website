import os
import django
import sys

# Step 1: Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Step 2: Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

# Step 3: Import your Product model
from products.models import Product

# Step 4: Create sample products
sample_products = [
    {
        "name": "Lavender",
        "price": 120.0,
        "category": "fragrant",
        "image_url": "https://example.com/lavender.jpg"
    },
    {
        "name": "Jasmine",
        "price": 150.0,
        "category": "fragrant",
        "image_url": "https://example.com/jasmine.jpg"
    },
    {
        "name": "Snake Plant",
        "price": 90.0,
        "category": "indoor",
        "image_url": "https://example.com/snakeplant.jpg"
    }
]

# Step 5: Insert into MongoDB
for prod in sample_products:
    p = Product(**prod)
    p.save()

print("✅ Sample products inserted into MongoDB.")
