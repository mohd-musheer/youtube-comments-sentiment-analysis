# YouTube Community Mental Health Analysis

An AI-powered web application that analyzes YouTube comments to generate a **mental health profile of a video’s community**.  
The system fetches real user comments, classifies them using a machine learning model, and presents aggregated insights through a modern web dashboard.

---

## 🚀 Live Demo
- 🌐 **Live App (Render):** [youtube-comments-sentiment-analysis-ai.onrender.com](https://youtube-comments-sentiment-analysis-ai.onrender.com)

- 🐳 **Docker Image:** [mohdmusheer/youtube-comments-sentiment-analysis](https://hub.docker.com/r/mohdmusheer/youtube-comments-sentiment-analysis)

---

## 🧠 What This Project Does

1. User inputs a **YouTube video URL**
2. System fetches **top English comments** using YouTube Data API v3
3. Each comment is analyzed by a trained ML model
4. Comments are classified into **7 mental health categories**
5. Results are aggregated into a **community mental health profile**
6. Dashboard displays:
   - Video title & thumbnail
   - Category-wise percentages
   - Recent comments with predicted labels

---

## 🧩 Mental Health Categories

- Normal  
- Anxiety  
- Depression  
- Stress  
- Bipolar  
- Personality Disorder  
- Suicidal  

> ⚠️ Low-confidence predictions are conservatively mapped to **Normal** to reduce misclassification risk.

---

## 🏗️ System Architecture

User → Web UI → FastAPI Backend
↓
YouTube Data API (comments)
↓
NLP Preprocessing (TF-IDF)
↓
Logistic Regression Classifier
↓
Aggregation & Visualization

yaml
Copy code

---

## 🤖 Machine Learning Model

- **Algorithm:** Logistic Regression  
- **Vectorization:** TF-IDF Vectorizer  
- **Framework:** scikit-learn  
- **Training Accuracy:** **77%**
- **Language:** English-only (filtered at API level)

The model was trained on a labeled mental health sentiment dataset and optimized for interpretability and safety.

---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI
- scikit-learn
- NLTK
- langdetect
- Google YouTube Data API v3

### Frontend
- HTML
- CSS
- JavaScript (Fetch API)

### DevOps
- Docker
- Render (Live Deployment)

---

## 🐳 Docker Usage

### Pull Image
```bash
docker pull mohdmusheer/youtube-comments-sentiment-analysis
Run Container
bash
Copy code
docker run -p 8000:8000 mohdmusheer/youtube-comments-sentiment-analysis
Then open:

arduino
Copy code
http://localhost:8000
⚙️ Local Setup
bash
Copy code
git clone https://github.com/<YOUR_USERNAME>/youtube-community-mental-health-analysis.git
cd youtube-community-mental-health-analysis
pip install -r requirements.txt
python -m uvicorn api.api:app --reload

```
👥 Team & Collaboration
This project was developed collaboratively by 4 contributors:

| Name | GitHub Profile |
| :--- | :--- |
| **Mohd Musheer** | [github.com/mohd-musheer](https://github.com/mohd-musheer) |
| **Abhisheek** | [github.com/Abhisheek34](https://github.com/Abhisheek34) |
| **Shaurya Singru** | [github.com/yashaur](https://github.com/yashaur) |
| **Shashwat V** | [github.com/Vork-Shashwat](https://github.com/Vork-Shashwat) |
****
```
Collaboration was managed using GitHub with distributed task ownership and shared code reviews.

📌 Use Cases
Community mental health analysis

Social media research

Academic projects

NLP & ML demonstrations

Ethical AI case studies

