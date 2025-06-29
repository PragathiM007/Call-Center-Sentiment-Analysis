# Call Center Sentiment Analysis using BERT + LSTM

## 🧠 Project Overview
This project leverages Natural Language Processing (NLP) and Machine Learning (ML) techniques to analyze customer service call transcripts. The primary goal is to identify key drivers of customer dissatisfaction and predict customer sentiment to improve agent training and operational efficiency in call centers.

Developed as a capstone for Bellevue University's DSC550 - Data Mining course, this project demonstrates a production-ready sentiment analysis pipeline using real-world call center data.

---

## 📁 Data Sources
- **Call Transcripts**: 3 JSON files from [Summa Linguae Technologies](https://summalinguae.com/data-sets/call-center-data-in-us-english/) — includes speaker IDs, timestamps, and transcribed text.
- **Metadata**: CSV file containing customer satisfaction (CSAT) scores, call reasons, durations, and customer locations.

---

## 🎯 Objectives
- Identify common themes driving negative customer sentiment.
- Predict CSAT scores from conversation transcripts using ML models.
- Improve agent training strategies through data-driven insights.
- Evaluate the impact of noise and speaker workload on customer satisfaction.

---

## 🛠️ Technologies Used
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- NLP: NLTK, SpaCy, BERT (DistilBERT), LSTM (TensorFlow/Keras)
- Topic Modeling: LDA (Gensim)
- Modeling: Linear Regression, Random Forest, BERT + LSTM
- Visualization: Matplotlib, Seaborn

---

## 🔍 Key Insights

### 💬 Topic Modeling Results (LDA)
- **Topic 1**: Billing issues (`"charge", "invoice", "dispute"`)
- **Topic 2**: Reservation errors (`"booking", "cancel", "date"`)
- **Topic 3**: Service outages (`"down", "fix", "refund"`)

### 📉 Model Performance
| Model            | MAE  | R² Score | Notes                                |
|------------------|------|----------|--------------------------------------|
| Linear Regression| 0.16 | -0.02     | Poor fit for sparse textual features |
| Random Forest    | 0.14 | 0.12      | Slight overfitting                   |
| **BERT + LSTM**  | 0.12 | **0.41**  | Best contextual prediction model     |

### 🔎 Other Insights
- Billing disputes & reservation issues drive 62% of negative sentiment.
- High noise levels in calls → 22% drop in CSAT scores.
- Speaker workload imbalance affects satisfaction: top 3 speakers handled 45% of calls.

---

## 🧪 Data Pipeline Summary

1. **Preprocessing**:
   - Removed non-speech and noise segments
   - Lemmatization, stopword removal, case normalization
2. **Feature Engineering**:
   - Sentiment score (DistilBERT)
   - Call duration
   - Speaker activity
3. **Modeling**:
   - Multiple regression and classification models
   - Final model: BERT embeddings + LSTM for contextual CSAT prediction

---

## 📈 Recommendations
- **Agent Training**: Focus on billing resolution workflows
- **Real-Time Alerts**: Flag noisy or high-risk calls for supervision
- **Dashboards**: Use Tableau or Plotly for ongoing CSAT and sentiment tracking
- **Model Improvements**: Test more advanced transformer models like RoBERTa

---

## 🚧 Limitations
- 23% missing CSAT data limited model accuracy.
- Noise overlap in transcripts affects sentiment clarity.
- Need for additional metadata (e.g., customer tier) to personalize insights.

---

## 📦 Folder Structure
PAPM_DSC550/
├── data/
│ ├── transcripts/*.json
│ └── metadata.csv
├── notebooks/
│ ├── eda.ipynb
│ ├── modeling.ipynb
├── outputs/
│ ├── visuals/
│ └── models/
├── README.md


---

## 👤 Author
**Pragathi Porawakara Arachchige**  
Bellevue University – DSC550 Data Mining  
[GitHub](https://github.com/PragathiM007)
