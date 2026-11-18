# ClauseGuard AI - Machine Learning Model Evaluation Report

**Model Architecture:** TF-IDF (1-2 N-grams, Sublinear TF) + Calibrated Multiclass Logistic Regression  
**Evaluated On:** 125 annotated legal clauses across 10 student contract domains  
**Evaluation Methodology:** 80/20 Stratified Holdout Split + 5-Fold Stratified Cross-Validation  

---

## 1. Summary Metrics

- **Holdout Test Accuracy:** `80.00%`
- **5-Fold Cross-Validation Mean Accuracy:** `67.20% (± 3.92%)`
- **Macro Average F1-Score:** `0.7914`
- **Weighted Average F1-Score:** `0.7977`

---

## 2. Detailed Classification Report

```
                         precision    recall  f1-score   support

    CONFIDENTIALITY_NDA       0.50      0.50      0.50         2
         DEPOSIT_REFUND       0.75      1.00      0.86         3
   DISPUTE_JURISDICTION       1.00      1.00      1.00         2
 FINANCIAL_PENALTY_BOND       1.00      0.67      0.80         3
  INTELLECTUAL_PROPERTY       1.00      1.00      1.00         2
    LIABILITY_INDEMNITY       1.00      0.67      0.80         3
RESTRICTIVE_NON_COMPETE       0.67      1.00      0.80         2
          STANDARD_FAIR       0.50      0.50      0.50         2
  STIPEND_PAYMENT_TERMS       1.00      0.67      0.80         3
     TERMINATION_NOTICE       0.75      1.00      0.86         3

               accuracy                           0.80        25
              macro avg       0.82      0.80      0.79        25
           weighted avg       0.83      0.80      0.80        25

```

---

## 3. Top Predictive N-Grams (Feature Importance by Class)

- **CONFIDENTIALITY_NDA:** confidential, years following, period years, student shall, information
- **DEPOSIT_REFUND:** refunded, room, hostel, caution, deposit
- **DISPUTE_JURISDICTION:** exclusive jurisdiction, disputes, jurisdiction, courts, disputes shall
- **FINANCIAL_PENALTY_BOND:** liquidated, liquidated damages, damages, rs, 000
- **INTELLECTUAL_PROPERTY:** existing, inventions, property, created, company
- **LIABILITY_INDEMNITY:** defend, incurred, disclaims, party, liability
- **RESTRICTIVE_NON_COMPETE:** competing, software, compete, non, months
- **STANDARD_FAIR:** agreement, relationship, writing, counterparts, parties
- **STIPEND_PAYMENT_TERMS:** hours, month, calendar, monthly, stipend
- **TERMINATION_NOTICE:** days written, terminate, written notice, written, notice

---
## 4. Verification and Separation of ML vs Rule Engine
- **Clause Categorization:** Predicted by TF-IDF statistical vectorizer and calibrated linear probability model.
- **Statutory / Student Protection Violations:** Evaluated by separate deterministic compliance engine (`rule_engine.py`) to prevent black-box hallucination.
