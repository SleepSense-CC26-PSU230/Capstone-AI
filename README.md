# SleepSense Flask API
**Team CC26-PSU230 | Coding Camp 2026 DBS Foundation**

REST API berbasis Flask untuk screening awal risiko stres dan gangguan tidur.  
Menggabungkan **TensorFlow model** (klasifikasi risiko) dengan **Gemini 2.0 Flash** via LangChain untuk respons empatik.

> ⚠️ Output adalah **screening awal**, bukan diagnosis medis.

---

## Tech Stack

| Komponen | Library / Tool |
|---|---|
| Web Framework | Flask 3.0.3 |
| ML Model | TensorFlow CPU 2.18.0 |
| Generative AI | Gemini 2.0 Flash (`langchain-google-genai` 2.0.11) |
| LLM Orchestration | LangChain |
| WSGI Server | Gunicorn 22.0.0 |
| Containerization | Docker + Docker Compose |

---

## Struktur Project

```
sleepsense_flask/
├── app/
│   └── main.py                     ← Flask API (endpoints, TF model, Gemini chain)
├── models/
│   ├── sleepsense_model.keras      ← TF model hasil training (tidak di-commit)
│   ├── scaler_params.json          ← StandardScaler mean & scale dari training
│   └── feature_meta.json           ← Metadata fitur (gender_map, feature_cols)
├── notebooks/
│   ├── SleepSense_Training.ipynb   ← Training pipeline (Google Colab)
│   └── predict_colab.ipynb         ← Inference demo standalone (Google Colab)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Endpoints

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/health` | Health check & info service |
| `POST` | `/predict` | Klasifikasi risiko dari TF model |
| `POST` | `/chat` | Saran empatik dari Gemini 2.0 Flash |
| `POST` | `/analyze` | Predict + Chat dalam satu request |

---

## Request & Response

### `POST /predict`

**Request Body:**
```json
{
  "age": 22,
  "gender": "Male",
  "sleep_duration_hours": 5.5,
  "sleep_quality_score": 4.0,
  "daily_screen_time_hours": 8.0,
  "pre_sleep_screen_time_hours": 2.5,
  "physical_activity_minutes": 15,
  "caffeine_intake_cups": 4,
  "mental_fatigue_score": 7.5,
  "notifications_received_per_day": 120
}
```

**Response Body:**
```json
{
  "risk_label": "At Risk",
  "risk_level": "Tinggi",
  "risk_probability": 0.8234,
  "summary": "Pola tidur dan screen time Anda memerlukan perhatian segera.",
  "disclaimer": "Ini adalah screening awal, BUKAN diagnosis medis."
}
```

**Risk Level Mapping:**

| Probabilitas | `risk_label` | `risk_level` |
|---|---|---|
| < 0.35 | `No Risk` | Rendah |
| 0.35 – 0.65 | `Moderate Risk` | Sedang |
| > 0.65 | `At Risk` | Tinggi |

---

### `POST /chat`

Menerima data yang sama seperti `/predict`, ditambah dua field hasil prediksi:

```json
{
  "age": 22,
  "gender": "Male",
  "sleep_duration_hours": 5.5,
  "...": "...",
  "risk_level": "Tinggi",
  "risk_probability": 0.8234
}
```

**Response Body:**
```json
{
  "message": "Halo! Berdasarkan data tidurmu, kamu perlu memperhatikan..."
}
```

---

### `POST /analyze`

Endpoint utama — menjalankan prediksi TF model **dan** Gemini dalam satu request.

**Request Body:** sama dengan `/predict` (tanpa perlu menyertakan `risk_level`).

**Response Body:**
```json
{
  "prediction": {
    "risk_label": "At Risk",
    "risk_level": "Tinggi",
    "risk_probability": 0.8234,
    "summary": "Pola tidur dan screen time Anda memerlukan perhatian segera.",
    "disclaimer": "Ini adalah screening awal, BUKAN diagnosis medis."
  },
  "advice": "Halo! Aku sangat memahami betapa lelahnya kamu saat ini..."
}
```

---

## Setup & Menjalankan

### Step 1 — Training Model (Google Colab)

1. Buka `notebooks/SleepSense_Training.ipynb` di Google Colab
2. Jalankan semua cell dari atas ke bawah
3. Download `sleepsense_flask_models.zip` yang di-generate di Cell terakhir
4. Extract dan letakkan file ke folder `models/`:

```
models/
├── sleepsense_model.keras
├── scaler_params.json
└── feature_meta.json
```

> Untuk demo inference tanpa server Flask, gunakan `notebooks/predict_colab.ipynb`.

---

### Step 2 — Mendapatkan Gemini API Key

1. Buka [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Login dengan akun Google
3. Klik **Create API Key**
4. Copy key untuk dipakai di Step 3

---

### Step 3 — Setup Environment

```bash
# Copy .env.example ke .env
cp .env.example .env

# Isi GEMINI_API_KEY
nano .env
```

Isi file `.env`:
```
GEMINI_API_KEY=AIza...
PORT=5000
```

> ⚠️ Jangan pernah commit file `.env` ke GitHub. File ini sudah masuk `.gitignore`.

---

### Step 4 — Jalankan dengan Docker

```bash
# Build dan jalankan container
docker compose up -d

# Cek status container
docker compose ps

# Lihat log real-time
docker compose logs -f
```

Container akan otomatis restart jika crash (`restart: unless-stopped`) dan memiliki health check setiap 30 detik ke endpoint `/health`.

---

### Step 5 — Test API

```bash
# Health check
curl http://localhost:5000/health

# Predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 22, "gender": "Male",
    "sleep_duration_hours": 5.5, "sleep_quality_score": 4.0,
    "daily_screen_time_hours": 8.0, "pre_sleep_screen_time_hours": 2.5,
    "physical_activity_minutes": 15, "caffeine_intake_cups": 4,
    "mental_fatigue_score": 7.5, "notifications_received_per_day": 120
  }'

# Analyze (Predict + Gemini sekaligus)
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "age": 22, "gender": "Male",
    "sleep_duration_hours": 5.5, "sleep_quality_score": 4.0,
    "daily_screen_time_hours": 8.0, "pre_sleep_screen_time_hours": 2.5,
    "physical_activity_minutes": 15, "caffeine_intake_cups": 4,
    "mental_fatigue_score": 7.5, "notifications_received_per_day": 120
  }'
```

---

## Docker Commands Lainnya

```bash
# Build ulang setelah ada perubahan kode
docker compose up -d --build

# Stop container
docker compose down

# Masuk ke dalam container
docker exec -it sleepsense-api bash

# Restart container
docker compose restart
```

> Model di-mount sebagai volume (`./models:/sleepsense/models:ro`), sehingga bisa update model tanpa perlu rebuild Docker image.

---

## Deploy ke Server (VPS/Cloud)

```bash
# Clone repository
git clone <repo-url>
cd sleepsense_flask

# Upload model dari lokal ke server
scp sleepsense_flask_models.zip user@server:/path/to/sleepsense_flask/
ssh user@server "cd /path/to/sleepsense_flask && unzip sleepsense_flask_models.zip -d models/"

# Setup environment
cp .env.example .env
nano .env  # isi GEMINI_API_KEY

# Build dan jalankan
docker compose up -d --build

# Verifikasi
curl http://YOUR_SERVER_IP:5000/health
```

---

## Catatan Teknis

**Custom TF Components** (didefinisikan di `app/main.py`, harus konsisten dengan training):
- `AttentionScaling` — Custom Layer: soft attention gate per fitur
- `FocalLoss` — Custom Loss: menangani class imbalance (γ=2.0, α=0.25)

**Gemini Integration:**
- Model: `gemini-2.0-flash`
- Library: `langchain-google-genai==2.0.11`
- Prompt dirancang untuk respons empatik dalam Bahasa Indonesia dengan 2–3 saran konkret

**Singleton Loading:**  
Model TF dan LangChain chain di-load sekali saat pertama kali dibutuhkan (`_model`, `_llm_chain`) untuk menghindari overhead di setiap request.
