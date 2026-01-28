# 🌟 SAP COACH - AI Coaching Assistant

**Revenue-focused MVP built in 2 weeks**

SAP Coach is an AI-powered coaching assistant that uses developmental stage theory (SAP - Stanfield's Axiom of Perpetuity) to provide personalized guidance.

---

## 🚀 **Quick Start**

### **1. Install Dependencies**

```bash
pip install -r requirements_sapcoach.txt
```

### **2. Set Up API Keys**

Get your OpenAI API key from https://platform.openai.com/api-keys

```bash
# Windows
set OPENAI_API_KEY=sk-your-key-here

# Mac/Linux
export OPENAI_API_KEY=sk-your-key-here
```

### **3. Run the App**

```bash
streamlit run sap_coach_app.py
```

Open your browser to http://localhost:8501

---

## 💰 **Pricing**

- **Free Trial:** 10 messages
- **Pro:** $29/month (unlimited messages)
- **Team:** $99/month (5 users)

---

## 📊 **What is SAP?**

SAP (Stanfield's Axiom of Perpetuity) is a 9-stage developmental framework that maps consciousness evolution:

0. **Plenara** - Primordial, unformed
1. **Spark** - Initial ignition
2. **Polarity** - Binary thinking
3. **Motion** - Action, execution
4. **Foundation** - Structure, stability
5. **Threshold** - Critical decision point
6. **Integration** - Nuanced thinking
7. **Illusion** - Reality testing
8. **Rigidity** - Crystallization (trap risk)
9. **Renewal** - Transcendence

SAP Coach diagnoses which stage a user is in and provides stage-appropriate guidance.

---

## 🎯 **Target Market**

- Life coaches
- Therapists
- Executive coaches
- Career counselors
- HR professionals
- Organizational development consultants

---

## 🛠️ **Tech Stack**

- **AI:** OpenAI GPT-4 API
- **Frontend:** Streamlit
- **Payments:** Stripe
- **Hosting:** Streamlit Cloud (free tier)

---

## 📁 **File Structure**

```
SAP_COACH/
├── sap_coach_mvp.py          # Core chatbot logic
├── sap_coach_app.py           # Streamlit web interface
├── sap_coach_payments.py      # Stripe integration
├── requirements_sapcoach.txt  # Dependencies
├── SAP_COACH_LAUNCH_GUIDE.md  # Complete launch plan
└── README_SAPCOACH.md         # This file
```

---

## 🚀 **Deployment**

### **Streamlit Cloud (Recommended)**

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repo
4. Set environment variables:
   - `OPENAI_API_KEY`
   - `STRIPE_SECRET_KEY` (when ready for payments)
5. Deploy!

### **Alternative: Heroku**

```bash
heroku create sap-coach
heroku config:set OPENAI_API_KEY=sk-...
git push heroku main
```

---

## 💡 **Revenue Timeline**

**Conservative:**
- Month 1: $145 MRR (5 customers)
- Month 3: $870 MRR (30 customers)
- Month 6: $2,900 MRR (100 customers)

**Optimistic:**
- Month 1: $290 MRR (10 customers)
- Month 3: $2,175 MRR (75 customers)
- Month 6: $8,700 MRR (300 customers)

---

## 📝 **Next Steps**

1. ✅ Get OpenAI API key
2. ✅ Test locally
3. ✅ Deploy to Streamlit Cloud
4. ✅ Get 5 beta testers
5. ✅ Set up Stripe
6. ✅ Launch!

See `SAP_COACH_LAUNCH_GUIDE.md` for complete 2-week plan.

---

## ⚠️ **Important Disclaimers**

- SAP Coach is an AI assistant, not a licensed therapist
- For educational and coaching purposes only
- Not a replacement for professional mental health care
- Consult licensed professionals for serious issues

---

## 📞 **Support**

- Email: support@sapcoach.com (set up later)
- Documentation: See LAUNCH_GUIDE.md
- Issues: GitHub Issues

---

## 📄 **License**

Proprietary - All rights reserved

---

**Built with ❤️ to help coaches help more people**

**Launch Date:** 2026-01-25  
**Status:** Ready for Beta Testing  
**Next Milestone:** First Paying Customer
