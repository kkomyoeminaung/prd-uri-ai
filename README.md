# PRD-URI AI v4
**Unified Relational Intelligence** — Gemini Free API · Browser-ready  
*Myo Min Aung, 2026*

---

## 🚀 Deploy လုပ်နည်း (20 မိနစ်)

### Step 1 — Gemini API Key ရယူပါ (အပြည့်အဝ အခမဲ့)
```
aistudio.google.com → Get API Key → Create API Key → Copy
```
Free tier: **15 req/min, 1500 req/day** — သာမာန်သုံးဖို့ လုံလောက်တယ်

---

### Step 2 — GitHub ထည့်ပါ
```bash
git init
git add .
git commit -m "PRD-URI AI v4"
git remote add origin https://github.com/YOUR_NAME/prd-uri-ai.git
git push -u origin main
```

---

### Step 3 — Railway (Backend) — အခမဲ့
1. railway.app → New Project → GitHub repo ရွေး
2. `backend/` folder ကို root set လုပ်ပါ
3. Environment Variables ထည့်ပါ:
   ```
   GEMINI_API_KEY = AIzaSy-xxxxxxxxxx
   PORT = 8000
   ```
4. Deploy → URL copy ပါ (e.g. `https://xxx.railway.app`)

---

### Step 4 — Vercel (Frontend) — အပြည့်အဝ အခမဲ့
1. vercel.com → New Project → GitHub repo ရွေး
2. `frontend/` folder ကို root set လုပ်ပါ
3. Environment Variables:
   ```
   NEXT_PUBLIC_API_URL = https://xxx.railway.app
   ```
4. Deploy → URL ရပြီ! 🎉

---

## ကုန်ကျစရိတ်

| Service | ကုန်ကျမှု |
|---------|---------|
| Gemini API | **$0** (1500 req/day အခမဲ့) |
| Railway | **$0** (free tier) |
| Vercel | **$0** (free forever) |
| **စုစုပေါင်း** | **$0/လ** |

---

## Local Run
```bash
cd backend && pip install -r requirements.txt
cp ../.env.example ../.env  # GEMINI_API_KEY ထည့်
uvicorn app.main:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev
```
