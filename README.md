# GardenEase

**AI-powered gardening platform** — a full-stack web app built with Django and MongoDB. GardenEase combines **smart plant identification** (computer vision with YOLO) with e‑commerce, labor booking, and community. Discover products, book garden services, connect with gardeners, and **identify plants from a single photo** in seconds.

---

## Features

- **User Authentication** — Sign up, login, and Google OAuth (social login).
- **Products** — Browse and add gardening products to cart (categories, cart stored in MongoDB).
- **Labor Booking** — Book garden labor (lawn mowing, pruning, watering, etc.) with time slots.
- **Community** — Create posts, reply, and share with other gardeners (suggestions, sharing, problems, design).
- **Plant Detection** — Upload a plant image and get the plant name and confidence score (see [Plant Detection](#plant-detection) below).

---

## Tech Stack

| Layer        | Technology                          |
|-------------|--------------------------------------|
| Backend     | Django 5.x                           |
| Database    | MongoDB (via MongoEngine ODM)        |
| Auth        | Django auth + Google OAuth (social-auth-app-django) |
| API / CORS  | django-cors-headers                  |
| Plant AI    | Ultralytics YOLO (PyTorch)           |
| Frontend    | Static HTML, CSS, JavaScript (served by Django) |

---

## Project Structure

```
GardenEase - main/
├── backend/                    # Django project
│   ├── backend/                # Project settings, urls, wsgi
│   ├── community/              # Community posts & replies
│   ├── labor/                  # Labor booking & reviews
│   ├── plant_detection/        # Plant identification (YOLO)
│   ├── plant_model/            # YOLO weights (best.pt) — not in repo
│   ├── products/               # Products & cart
│   ├── userauth/               # Auth & Google OAuth
│   ├── media/                  # Uploaded files (ignored by git)
│   ├── manage.py
│   └── requirements.txt
├── frontend/                   # Static frontend (HTML/CSS/JS)
│   ├── Index/
│   ├── Cart/
│   ├── Garden Community/
│   ├── Labor/
│   ├── Our Products/
│   ├── Plantidentification/
│   └── loginSignup/
├── .gitignore
└── README.md
```

---

## Prerequisites

- **Python** 3.10 or higher  
- **MongoDB** running locally (default: `localhost:27017`)  
- **(Optional)** CUDA for GPU-accelerated plant detection (CPU works too)

---

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/GardenEase.git
   cd GardenEase
   ```

2. **Create and activate a virtual environment**
   ```bash
   cd backend
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **MongoDB**
   - Install and start MongoDB.
   - The app uses database name `gardenease_db` and connects to `localhost:27017` (see `backend/backend/settings.py`).

5. **Plant detection model**
   - The app expects a YOLO model at `backend/plant_model/best.pt`.
   - This file is **not** in the repo (too large for GitHub). Either:
     - Place your trained `best.pt` there, or  
     - Train a model with Ultralytics on your plant dataset and export `best.pt` to that path.

6. **Optional: Google OAuth**
   - Set your Google OAuth client ID and secret in `backend/backend/settings.py` (or move them to environment variables for production).

---

## How to Run

From the **backend** directory (where `manage.py` is):

```bash
cd backend
python manage.py runserver
```

Then open **http://127.0.0.1:8000/** in your browser. The Django server serves both the API and the frontend static files.

---

## Try the Plant Detection demo (CV)

To show that **Computer Vision** is implemented in this project:

1. Run the server (`cd backend` then `python manage.py runserver`).
2. Open **http://127.0.0.1:8000/plant/** in your browser.
3. Click **Upload Image** and choose a plant photo (JPG/PNG).
4. The app sends the image to the backend; the YOLO model runs inference and returns the **predicted plant class** and **confidence score**.
5. The result is displayed with a **confidence bar** and a **"Powered by YOLO (Computer Vision)"** label on the page.

This demonstrates the full CV pipeline: image upload → backend inference (YOLO) → JSON response → UI update with class and confidence.

---

## Plant Detection

Plant detection is one of the core features of GardenEase: users can upload a photo of a plant and get an identification (class name) and a confidence score.

### Overview

- **Technology:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (PyTorch).
- **Model:** A custom-trained YOLO model whose weights are stored in `backend/plant_model/best.pt`. The app loads this model once at startup and reuses it for every request.

### Flow

1. **Frontend**  
   - User clicks “Upload Image” on the Plant Identification page (`/plant/`).  
   - They select an image file (e.g. JPG, PNG).  
   - The frontend sends a `POST` request to `/plant/detect/` with the image in a `multipart/form-data` body (field name: `image`).

2. **Backend**  
   - Django receives the file in `plant_detection/views.py`.  
   - The image is saved temporarily under `media/temp/`.  
   - The same view calls the already-loaded YOLO model with `model.predict(full_file_path)`.  
   - From the result, it reads the top predicted class name and confidence.  
   - The temporary file is deleted.  
   - The view returns a JSON response: `{"class_name": "...", "confidence": 0.95}` (or an error).

3. **Frontend**  
   - The script in `plant_detection/templates/plant_detection/script.js` receives the JSON and updates the UI with the detected plant name and confidence (e.g. “Confidence: 95.00%”).

### API

| Method | Endpoint       | Description |
|--------|----------------|-------------|
| GET    | `/plant/`      | Renders the plant identification page. |
| POST   | `/plant/detect/` | Expects `image` (file). Returns `{ "class_name": "...", "confidence": float }` or `{ "error": "..." }`. |

### Code locations

- **Views & API:** `backend/plant_detection/views.py` — `detect_plant` (POST), `plant_page` (GET), and helper `detect_plant_from_path` for server-side path-based detection.
- **URLs:** `backend/plant_detection/urls.py` — routes for `''` (page) and `'detect/'` (API).
- **Model path:** `backend/plant_model/best.pt` (loaded once in `views.py`).
- **Frontend:** `backend/plant_detection/templates/plant_detection/index.html` and `script.js` (upload button, fetch to `/plant/detect/`, display of result).

### Model and dataset

- **Weights:** The file `best.pt` is produced by training (or exporting) a YOLO model on a plant dataset. It is not included in the repo; you must add it under `backend/plant_model/` (or change `MODEL_PATH` in `views.py`).
- **Dataset:** Any dataset used for training (e.g. in `Plant-Detection-Model-using-Yolo-main/` or other folders) should stay local and is listed in `.gitignore` so it is not uploaded to GitHub.

### Implementation summary (CV in code)

The plant detection API is implemented in **`backend/plant_detection/views.py`**:

- **Model loading:** The YOLO model is loaded once at module import from `backend/plant_model/best.pt`.
- **Endpoint `detect_plant`:** Accepts `POST` with an `image` file, saves it temporarily, runs `model.predict(...)`, reads the top class and confidence from the result, then returns JSON and deletes the temp file.
- **Helper `detect_plant_from_path`:** For server-side use; takes an image path and returns `class_name` and `confidence`.

Example response from `/plant/detect/`:

```json
{ "class_name": "Monstera Deliciosa", "confidence": 0.92 }
```

The frontend (`plant_detection/templates/plant_detection/`) shows a **confidence bar** and a **"Powered by YOLO (Computer Vision)"** label so viewers can see that the CV pipeline is integrated end-to-end.

---

## Environment & Security (Production)

- Do **not** commit real API keys or secrets. Prefer environment variables (e.g. `SECRET_KEY`, Google OAuth client ID/secret) and a `.env` file that is in `.gitignore`.
- Set `DEBUG = False` and configure `ALLOWED_HOSTS` and static/media serving for production.

---

## License & Purpose

This project is **for portfolio and educational purposes** — developed to demonstrate full-stack development, AI/ML integration (plant identification), and modern web practices. Feel free to explore the code and use it as a reference. If you build on it or use parts of it, attribution is appreciated.

*Not intended for commercial use without permission.*
