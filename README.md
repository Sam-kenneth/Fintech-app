# 🚀 AI-Powered Fintech Risk Analyzer

A sophisticated risk assessment engine that combines traditional financial data (GST, PAN, ITR) with AI-driven narrative analysis. Built for speed and accuracy using RAG (Retrieval-Augmented Generation).

## ✨ Key Features
* **Multi-Doc Analysis:** Process PDFs and images via OCR to extract financial health.
* **Narrative Insight:** Uses AI to detect "Intent Risk" and inconsistencies in borrower statements.
* **API Integration:** Mocked support for GST, PAN, and Credit Bureau data.
* **Cloud Ready:** Optimized for Vercel Serverless deployment.

## 🛠️ Tech Stack
* **Backend:** Python (Flask)
* **AI Engine:** Groq / Llama 3 (via API)
* **Vector DB:** FAISS (for document retrieval)
* **Embeddings:** HuggingFace
* **Frontend:** Simple HTML5/JS Dashboard

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.9+
* A **Groq API Key** (Get it free at [console.groq.com](https://console.groq.com))

### 2. Installation
```bash
# Clone the repo
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

3. Environment Variables
Create a .env file in the root or add these to your system:

GROQ_API_KEY=your_key_here

4. Run Locally

Bash
python api/index.py
Open http://localhost:5000 in your browser.