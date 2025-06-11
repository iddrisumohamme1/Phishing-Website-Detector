# Phishing Website Detector

A web application that uses machine learning and rule-based analysis to detect phishing websites. Enter a URL to check its risk level and receive security recommendations.

## Features

- **Hybrid Detection:** Combines rule-based heuristics and a trained ML model (Random Forest) for high accuracy.
- **User-Friendly UI:** Clean, responsive web interface for easy URL checking.
- **Batch Analysis:** Analyze multiple URLs at once (API support).
- **Security Tips:** Educates users on identifying phishing websites.

## How It Works

1. **User submits a URL** via the web form.
2. **Backend (Flask)** extracts features from the URL and runs both rule-based and ML analysis.
3. **Results** are displayed with risk score, warnings, and recommendations.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/iddrisumohamme1/Phishing-Website-Detector.git
   cd "Phishing Website Detector"
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **(Optional) Add your own dataset:**

   - Place your CSV file (with URLs and labels) in the project directory.
   - Update the filename in `ml_model.py` if needed.
   - Re-train the model using `ml_model.py` using `python ml_model.py`.

### Running the App

1. **Start the Flask server:**

   ```sh
   python app.py
   ```

2. **Open the web interface:**

   - Go to [http://localhost:5000](http://localhost:5000) in your browser.

## File Structure

- `app.py` — Flask backend, API endpoints, and hybrid detection logic
- `ml_model.py` — Machine learning model, feature extraction, and training
- `index.html` — Main web UI
- `style.css` — Styles for the web UI
- `script.js` — Frontend logic (fetch, display results)
- `requirements.txt` — Python dependencies
- `models/` — Saved ML model and feature names
- `PhiUSIIL_Phishing_URL_Dataset.csv` — Example dataset (replace with your own for production)

## API Endpoints

- `POST /api/check` — Analyze a single URL (expects `{ "url": "..." }`)
- `POST /analyze` — (Legacy) Analyze a single URL
- `POST /batch-analyze` — Analyze multiple URLs (expects `{ "urls": [ ... ] }`)
- `GET /health` — Health check
- `GET /model-info` — Get ML model info
- `POST /retrain-model` — Retrain the ML model

## Customization

- **Dataset:** Replace `PhiUSIIL_Phishing_URL_Dataset.csv` with your own for better accuracy.
- **Model:** Tweak feature extraction or ML parameters in `ml_model.py`.

## Browser Extension (Optional)

You can create a browser extension popup using a mini version of the form. See the codebase or ask for a template.

## License

MIT License

---

Created by Mohammed Luriwie Iddrisu
