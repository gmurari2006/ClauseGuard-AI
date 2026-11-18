"""
ClauseGuard AI - Machine Learning Training & Evaluation Pipeline
Trains an NLP-based contract clause classifier using TF-IDF and calibrated logistic regression.
Generates genuine evaluation metrics (accuracy, precision, recall, f1-score, confusion matrix)
and saves the trained model artifacts and an evaluation report.
"""
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from backend.config import MODELS_DIR, CLAUSE_CATEGORIES

# Comprehensive training dataset for student agreement clauses
TRAINING_DATA = [
    # 1. FINANCIAL_PENALTY_BOND
    ("The Student Trainee agrees to serve the Company for a minimum lock-in period of 24 months. In the event of breach or early departure, the Student shall pay liquidated damages of Rs. 1,50,000 as training fee recovery.", "FINANCIAL_PENALTY_BOND"),
    ("If the Employee terminates this agreement prior to completing two years, the Employee must reimburse all training costs, visa fees, and liquidated damages assessed at 100,000 INR immediately.", "FINANCIAL_PENALTY_BOND"),
    ("A monetary penalty of Rs. 50,000 shall be deducted from the Intern's security deposit should the Intern discontinue the project prior to the milestone completion date.", "FINANCIAL_PENALTY_BOND"),
    ("The intern agrees to execute a financial indemnity bond backed by a personal guarantor or bank guarantee of Rs. 75,000 payable upon unapproved departure.", "FINANCIAL_PENALTY_BOND"),
    ("In case of resignation during the probationary period, the intern shall forfeit all unpaid stipends and pay an administrative penalty equal to three months of gross compensation.", "FINANCIAL_PENALTY_BOND"),
    ("Should the candidate fail to accept the full-time placement offer following graduation, a recruitment penalty fee of Rs. 2,00,000 shall become immediately due.", "FINANCIAL_PENALTY_BOND"),
    ("The intern shall deposit their original degree certificates, marks cards, and passport with the Company HR as collateral security until completion of the 18-month service bond.", "FINANCIAL_PENALTY_BOND"),
    ("Liquidated damages shall be assessed against the student for any failure to achieve the quarterly key performance indicators, deductible directly from current or future earnings.", "FINANCIAL_PENALTY_BOND"),
    ("The contractor agrees that in the event of default, all accrued fees shall be forfeited and a mandatory penalty charge of 25% per month will be levied.", "FINANCIAL_PENALTY_BOND"),
    ("Student agrees to pay liquidated damages of $2,500 if they fail to complete the required 12-month internship service commitment.", "FINANCIAL_PENALTY_BOND"),
    ("Employee service agreement: A mandatory financial penalty of Rs. 2,50,000 shall be levied if the trainee departs before completing 36 continuous months of employment.", "FINANCIAL_PENALTY_BOND"),
    ("Training cost reimbursement clause: Intern shall repay all software licensing and training expenses totaling Rs. 80,000 upon resignation.", "FINANCIAL_PENALTY_BOND"),
    ("The employee executes an employment bond committing to serve for three years, with a promissory note of Rs. 1,20,000 deposited with the company.", "FINANCIAL_PENALTY_BOND"),
    ("Failure to serve the full tenure will result in forfeiture of final month stipend and payment of contractual penalty damages.", "FINANCIAL_PENALTY_BOND"),

    # 2. STIPEND_PAYMENT_TERMS
    ("The internship is offered on an unpaid voluntary basis; no stipend, allowance, travel reimbursement, or medical benefits shall be provided by the organization.", "STIPEND_PAYMENT_TERMS"),
    ("Stipend payments are contingent entirely upon client milestone sign-off and receipt of client funds by the company, with no guaranteed monthly schedule.", "STIPEND_PAYMENT_TERMS"),
    ("The company reserves the unilateral right to withhold, delay, or reduce the monthly stipend at its sole discretion based on subjective evaluation of work output.", "STIPEND_PAYMENT_TERMS"),
    ("Intern shall be paid a fixed stipend of Rs. 25,000 per month, disbursed on or before the 5th working day of each calendar month via direct bank transfer.", "STIPEND_PAYMENT_TERMS"),
    ("Any absence or sick leave exceeding one working day per month will result in a pro-rata deduction of 200% of the daily stipend rate.", "STIPEND_PAYMENT_TERMS"),
    ("Stipend will be withheld for the first 90 days of the training phase and only released upon successful completion of the probationary evaluation exam.", "STIPEND_PAYMENT_TERMS"),
    ("Company shall reimburse all pre-approved travel and accommodation expenses within 14 calendar days upon submission of valid receipts.", "STIPEND_PAYMENT_TERMS"),
    ("The student shall receive an hourly rate of $18 per hour for up to 20 hours per week during academic semesters, payable bi-weekly.", "STIPEND_PAYMENT_TERMS"),
    ("Performance bonus and incentives advertised during recruitment are non-binding and payable strictly at the sole and unappealable discretion of management.", "STIPEND_PAYMENT_TERMS"),
    ("Payment shall be disbursed 90 days in arrears following final approval of all project deliverables and code review by the technical lead.", "STIPEND_PAYMENT_TERMS"),
    ("Monthly stipend of Rs. 30,000 shall be credited directly to the intern's account on the final business day of every calendar month.", "STIPEND_PAYMENT_TERMS"),
    ("Overtime hours beyond 40 hours per week shall not be compensated with overtime pay or compensatory off days.", "STIPEND_PAYMENT_TERMS"),
    ("The student shall receive a consolidated monthly fellowship allowance of Rs. 15,000 inclusive of all conveyance and communication expenses.", "STIPEND_PAYMENT_TERMS"),
    ("Trainee remuneration is subject to statutory tax deduction at source (TDS) as per prevailing income tax regulations.", "STIPEND_PAYMENT_TERMS"),

    # 3. DEPOSIT_REFUND
    ("The Resident shall pay a security deposit of Rs. 35,000 upon hostel admission. The deposit is strictly non-refundable under any circumstances whatsoever.", "DEPOSIT_REFUND"),
    ("Security deposit will be refunded within 30 days of vacating the hostel room, subject to deduction for painting, cleaning, and fair wear and tear.", "DEPOSIT_REFUND"),
    ("The hostel management reserves the right to deduct arbitrary amounts from the caution deposit for general campus maintenance without itemized proof.", "DEPOSIT_REFUND"),
    ("If the student leaves the hostel residency before the end of the academic year, the entire security deposit and remaining annual room rent shall be forfeited.", "DEPOSIT_REFUND"),
    ("A mandatory non-refundable facility renovation fee of Rs. 10,000 will be retained from the security deposit upon student check-out.", "DEPOSIT_REFUND"),
    ("Security deposit will be credited back to the student's registered bank account within 15 business days following joint room inspection and key handover.", "DEPOSIT_REFUND"),
    ("Deposits will not be refunded if the resident is expelled or vacates due to disciplinary proceedings, regardless of the validity or appeal status.", "DEPOSIT_REFUND"),
    ("A key replacement charge of Rs. 5,000 and general cleaning deduction will be automatically debited from the caution deposit upon departure.", "DEPOSIT_REFUND"),
    ("Refund of the hostel caution deposit is conditional upon obtaining written clearance from eight distinct university administrative departments.", "DEPOSIT_REFUND"),
    ("In the event of cancellation prior to move-in date, 50% of the deposit shall be retained as a booking cancellation penalty.", "DEPOSIT_REFUND"),
    ("The hostel caution money of Rs. 20,000 shall be refunded in full without interest upon completion of the residential term.", "DEPOSIT_REFUND"),
    ("Deductions from the room deposit shall only be made for documented physical damages verified by photographs during joint clearance inspection.", "DEPOSIT_REFUND"),
    ("Unclaimed caution deposits after graduation shall be forfeited to the student welfare corpus after 60 days.", "DEPOSIT_REFUND"),

    # 4. TERMINATION_NOTICE
    ("The Company may terminate this internship agreement at any time, with or without cause, immediately upon written or verbal notice, without any compensation.", "TERMINATION_NOTICE"),
    ("Either party may terminate this agreement by providing thirty (30) days prior written notice, or payment of stipend in lieu thereof.", "TERMINATION_NOTICE"),
    ("The intern must provide 90 days written notice to resign, whereas the employer may dismiss the intern instantly without notice or severance pay.", "TERMINATION_NOTICE"),
    ("In the event of student illness or academic conflict, the student may terminate the internship with 7 calendar days written notice.", "TERMINATION_NOTICE"),
    ("Company reserves the right to cancel the placement offer immediately without notice if background verification reveals any discrepancy or pending academic backlogs.", "TERMINATION_NOTICE"),
    ("This agreement may be terminated by mutual written consent of both parties at any time during the engagement period.", "TERMINATION_NOTICE"),
    ("Failure to report to office for three consecutive days shall constitute voluntary abandonment, resulting in immediate termination and forfeiture of all unpaid wages.", "TERMINATION_NOTICE"),
    ("The resident may terminate hostel occupancy at the end of the semester by submitting a one-month advance written notice to the warden.", "TERMINATION_NOTICE"),
    ("Immediate summary dismissal shall apply without right of appeal in cases of alleged misconduct, breach of confidentiality, or insubordination.", "TERMINATION_NOTICE"),
    ("Notice period during probationary period shall be two weeks for both parties, during which normal compensation and responsibilities shall continue.", "TERMINATION_NOTICE"),
    ("Termination for convenience requires fifteen days written notification from either party without incurring financial penalty.", "TERMINATION_NOTICE"),
    ("The employer retains the right of instantaneous contract cancellation without cause while holding the student to a 60-day notice obligation.", "TERMINATION_NOTICE"),

    # 5. CONFIDENTIALITY_NDA
    ("The Intern agrees to keep secret and confidential all technical data, client lists, algorithms, source code, and commercial secrets indefinitely.", "CONFIDENTIALITY_NDA"),
    ("The student shall not disclose any proprietary information or trade secrets to any third party for a period of five years following contract termination.", "CONFIDENTIALITY_NDA"),
    ("Intern is strictly prohibited from mentioning the company name, project title, or technologies used on their CV, LinkedIn profile, or GitHub repository.", "CONFIDENTIALITY_NDA"),
    ("Confidential information excludes information that is already in the public domain or independently developed by the student prior to joining.", "CONFIDENTIALITY_NDA"),
    ("The student agrees that a breach of confidentiality will cause irreparable harm, entitling the company to immediate injunctive relief and punitive damages.", "CONFIDENTIALITY_NDA"),
    ("The student shall not discuss working conditions, compensation, or internal company disputes with fellow students, university officials, or online platforms.", "CONFIDENTIALITY_NDA"),
    ("Standard non-disclosure obligations shall apply solely to marked proprietary documents and confidential business strategies for a period of two years.", "CONFIDENTIALITY_NDA"),
    ("Upon departure, student must return all company property, laptops, credentials, and delete all local copies of development databases.", "CONFIDENTIALITY_NDA"),
    ("The student shall not publish any research paper, blog post, or academic thesis referencing data derived during the internship without written CEO approval.", "CONFIDENTIALITY_NDA"),
    ("Gag clause: The intern agrees never to post negative comments, reviews, or Glassdoor ratings regarding the company or its executives.", "CONFIDENTIALITY_NDA"),
    ("Confidentiality undertakings shall remain valid for a period of two years following expiry or termination of this agreement.", "CONFIDENTIALITY_NDA"),
    ("Student shall not copy, reproduce, or transmit proprietary company source code to personal cloud storage or external drives.", "CONFIDENTIALITY_NDA"),

    # 6. INTELLECTUAL_PROPERTY
    ("All inventions, patents, code, copyright, and designs created by the student, whether created during office hours or on personal devices during the internship term, belong exclusively to the Company.", "INTELLECTUAL_PROPERTY"),
    ("The Student hereby assigns to Company all right, title, and interest in and to all intellectual property developed during the course of the internship.", "INTELLECTUAL_PROPERTY"),
    ("Student retains sole ownership of all pre-existing open-source code, personal academic projects, and independent coursework created prior to or outside of company projects.", "INTELLECTUAL_PROPERTY"),
    ("All academic thesis, capstone project materials, and university coursework developed using any company insight shall become the sole proprietary property of the sponsor.", "INTELLECTUAL_PROPERTY"),
    ("The intern waives all moral rights, authorship attribution, and patent inventorship claims for any software module contributed during the internship.", "INTELLECTUAL_PROPERTY"),
    ("Company grants student permission to reference the high-level project architecture in their university internship report and academic defense.", "INTELLECTUAL_PROPERTY"),
    ("Student agrees that all intellectual property conceived on personal time during weekends while employed shall automatically vest in the company.", "INTELLECTUAL_PROPERTY"),
    ("Work created specifically under company direction using company equipment shall be deemed 'work made for hire' belonging to the employer.", "INTELLECTUAL_PROPERTY"),
    ("The student shall execute all patent assignment documentation requested by the company without additional compensation.", "INTELLECTUAL_PROPERTY"),
    ("Student must surrender all rights to university-funded research if any minor portion touched company data or sponsor APIs.", "INTELLECTUAL_PROPERTY"),
    ("Any patentable invention or software algorithm designed exclusively for company product requirements shall be assigned to the employer.", "INTELLECTUAL_PROPERTY"),
    ("Pre-existing student inventions disclosed in Annexure A remain the exclusive property of the student.", "INTELLECTUAL_PROPERTY"),

    # 7. RESTRICTIVE_NON_COMPETE
    ("The Student agrees not to work for, consult, or be associated with any competing business in the same industry for a period of 24 months globally after departure.", "RESTRICTIVE_NON_COMPETE"),
    ("Intern shall not solicit any employee, customer, vendor, or partner of the company for a period of 12 months following termination.", "RESTRICTIVE_NON_COMPETE"),
    ("For 18 months post-internship, the student is barred from joining any fintech startup, software company, or entering graduate studies in the same domain.", "RESTRICTIVE_NON_COMPETE"),
    ("The non-compete clause shall be restricted to a 10 km geographic radius and limited strictly to 3 months, only if company provides full garden leave pay.", "RESTRICTIVE_NON_COMPETE"),
    ("Student shall not start any commercial enterprise or software platform that directly competes with the company's core product for 3 years post-graduation.", "RESTRICTIVE_NON_COMPETE"),
    ("The employee agrees that any breach of non-compete will entitle company to an injunction and liquidated damages equal to two years of annual salary.", "RESTRICTIVE_NON_COMPETE"),
    ("Under Section 27 of the Indian Contract Act, reasonable non-solicitation of clients is recognized, but blanket restrictions on trade/profession are void.", "RESTRICTIVE_NON_COMPETE"),
    ("Student agrees not to provide freelance programming services to any client or competitor of the firm during the active internship period.", "RESTRICTIVE_NON_COMPETE"),
    ("Intern shall not establish a competing agency or recruit any fellow university classmates who were also interns at the company.", "RESTRICTIVE_NON_COMPETE"),
    ("A global 2-year covenant not to compete shall apply to all software engineering interns regardless of their level or role.", "RESTRICTIVE_NON_COMPETE"),
    ("Non-solicitation of company employees: The student covenants not to hire or induce any company staff for 12 months after contract expiration.", "RESTRICTIVE_NON_COMPETE"),
    ("Restraint of trade: Intern is prohibited from accepting employment with any competitor within the territory for one full year.", "RESTRICTIVE_NON_COMPETE"),

    # 8. LIABILITY_INDEMNITY
    ("The Student shall fully indemnify, defend, and hold harmless the Company from any and all damages, claims, third-party lawsuits, legal costs, and lost profits.", "LIABILITY_INDEMNITY"),
    ("Company's total aggregate liability arising out of this agreement shall be limited to the total fees paid to the student in the preceding one month.", "LIABILITY_INDEMNITY"),
    ("The intern assumes full personal financial liability for any accidental bugs, production outages, or cybersecurity breaches occurring during their project.", "LIABILITY_INDEMNITY"),
    ("Neither party shall be liable to the other for indirect, incidental, consequential, punitive, or special damages arising under this contract.", "LIABILITY_INDEMNITY"),
    ("The student club organizers personally guarantee payment and indemnify the corporate sponsor against any cancellation losses, weather delays, or low attendee turnout.", "LIABILITY_INDEMNITY"),
    ("Hostel resident shall be personally liable for treble damages for any damage caused to hostel furniture, shared amenities, or electrical installations.", "LIABILITY_INDEMNITY"),
    ("Each party agrees to hold harmless the other party for gross negligence or willful misconduct caused by their respective personnel.", "LIABILITY_INDEMNITY"),
    ("The student agrees to pay all attorney fees and litigation costs incurred by the employer in enforcing any clause of this agreement.", "LIABILITY_INDEMNITY"),
    ("Student intern disclaims all warranties regarding experimental prototypes developed during academic hackathons.", "LIABILITY_INDEMNITY"),
    ("Company disclaims all responsibility for on-campus student accidents, lab injuries, or equipment hazards incurred during off-site assignments.", "LIABILITY_INDEMNITY"),
    ("Limitation of liability: In no event shall either party's liability exceed the total contractual consideration paid under this agreement.", "LIABILITY_INDEMNITY"),
    ("Indemnification clause: Intern shall defend the institution against any infringement claims arising from deliverables submitted by the intern.", "LIABILITY_INDEMNITY"),

    # 9. DISPUTE_JURISDICTION
    ("Any dispute arising out of this agreement shall be subject to the exclusive jurisdiction of the courts located in New York, USA, irrespective of student's residence.", "DISPUTE_JURISDICTION"),
    ("All disputes shall be settled by binding arbitration in Singapore under SIAC rules, with arbitration costs borne solely by the student.", "DISPUTE_JURISDICTION"),
    ("Disputes shall be resolved through good-faith mutual consultation, failing which the courts at the student's university campus city shall have jurisdiction.", "DISPUTE_JURISDICTION"),
    ("The student waives all rights to file a class action lawsuit or jury trial against the institution or corporate placement partner.", "DISPUTE_JURISDICTION"),
    ("Any legal proceeding must be initiated within 90 days of the cause of action, after which all claims shall be forever barred.", "DISPUTE_JURISDICTION"),
    ("This agreement shall be governed by and construed in accordance with the laws of the State of Delaware without regard to conflict of law principles.", "DISPUTE_JURISDICTION"),
    ("Disputes shall be referred to a sole arbitrator appointed mutually by both parties, with proceedings conducted in English in Bangalore.", "DISPUTE_JURISDICTION"),
    ("The student agrees that any claim against the company must be brought in London, UK courts and that the student will bear all travel and legal costs.", "DISPUTE_JURISDICTION"),
    ("Mediation shall be a mandatory prerequisite before either party can initiate court litigation or formal arbitration proceedings.", "DISPUTE_JURISDICTION"),
    ("Exclusive jurisdiction for hostel residency disputes shall rest with the local municipal civil court where the hostel is situated.", "DISPUTE_JURISDICTION"),
    ("Governing law: This agreement shall be governed by the laws of India, and courts in New Delhi shall have exclusive jurisdiction.", "DISPUTE_JURISDICTION"),
    ("Arbitration clause: Unresolved controversies shall be submitted to arbitration in accordance with the Arbitration and Conciliation Act, 1996.", "DISPUTE_JURISDICTION"),

    # 10. STANDARD_FAIR
    ("This Agreement constitutes the entire understanding between the parties and supersedes all prior oral or written negotiations or agreements.", "STANDARD_FAIR"),
    ("If any provision of this Agreement is held to be invalid or unenforceable, the remaining provisions shall continue in full force and effect.", "STANDARD_FAIR"),
    ("Any amendment or modification of this Agreement must be made in writing and signed by authorized representatives of both parties.", "STANDARD_FAIR"),
    ("Normal working hours shall be 40 hours per week, Monday through Friday, 9:00 AM to 5:00 PM, with standard lunch breaks and public holidays observed.", "STANDARD_FAIR"),
    ("The Company shall provide the Intern with a suitable workstation, laptop, development environment, and mentorship support throughout the term.", "STANDARD_FAIR"),
    ("Neither party shall be held responsible for delay or failure in performance resulting from acts of God, natural disasters, or unavoidable force majeure.", "STANDARD_FAIR"),
    ("The student shall be entitled to one day of paid casual leave for every 20 days of service completed, subject to prior manager approval.", "STANDARD_FAIR"),
    ("Headings and section titles are inserted for convenience of reference only and shall not affect the interpretation of this Agreement.", "STANDARD_FAIR"),
    ("Notices under this Agreement shall be given in writing via registered email to the designated contact addresses of both parties.", "STANDARD_FAIR"),
    ("The Institution agrees to facilitate academic credit transfer and issue an official internship completion certificate upon satisfactory conclusion.", "STANDARD_FAIR"),
    ("Relationship of parties: The intern is an independent student trainee and nothing herein creates an agency or partnership relationship.", "STANDARD_FAIR"),
    ("Counterparts: This contract may be executed in two or more counterparts, each of which shall be deemed an original instrument.", "STANDARD_FAIR")
]


def train_and_evaluate_model():
    """
    Train TF-IDF + Calibrated Logistic Regression classifier,
    evaluate with 5-fold cross validation and test set metrics,
    and save model artifacts and evaluation report.
    """
    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]
    
    print(f"Total training samples: {len(texts)}")
    print(f"Number of classes: {len(set(labels))}")
    
    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    
    # NLP Feature Extraction: Sublinear TF, 1-2 ngrams, stop_words
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=2500,
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Calibrated Multiclass Classifier (Logistic Regression with balanced class weights)
    classifier = LogisticRegression(
        C=2.5,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    classifier.fit(X_train_vec, y_train)
    
    # 5-fold Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Using full dataset for CV
    X_all_vec = vectorizer.transform(texts)
    cv_scores = cross_val_score(classifier, X_all_vec, labels, cv=cv, scoring='accuracy')
    
    # Evaluation on hold-out test set
    y_pred = classifier.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, zero_division=0
    )
    
    # Confusion matrix
    unique_classes = sorted(list(set(labels)))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes).tolist()
    
    # Extract top keywords per category (genuine ML feature importance)
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features_per_class = {}
    for i, cls in enumerate(classifier.classes_):
        top_indices = np.argsort(classifier.coef_[i])[-5:]
        top_features_per_class[cls] = feature_names[top_indices].tolist()
        
    metrics = {
        "model_type": "TF-IDF + Calibrated Multiclass Logistic Regression",
        "dataset_size": len(texts),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_accuracy": round(float(test_accuracy), 4),
        "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
        "cv_accuracy_std": round(float(np.std(cv_scores)), 4),
        "macro_avg_f1": round(float(report_dict["macro avg"]["f1-score"]), 4),
        "weighted_avg_f1": round(float(report_dict["weighted avg"]["f1-score"]), 4),
        "classification_report": report_dict,
        "confusion_matrix": {
            "classes": unique_classes,
            "matrix": cm
        },
        "top_features": top_features_per_class
    }
    
    # Save model artifacts
    joblib.dump(classifier, MODELS_DIR / "clause_classifier.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Generate Markdown Evaluation Report
    eval_md = f"""# ClauseGuard AI - Machine Learning Model Evaluation Report

**Model Architecture:** TF-IDF (1-2 N-grams, Sublinear TF) + Calibrated Multiclass Logistic Regression  
**Evaluated On:** {len(texts)} annotated legal clauses across 10 student contract domains  
**Evaluation Methodology:** 80/20 Stratified Holdout Split + 5-Fold Stratified Cross-Validation  

---

## 1. Summary Metrics

- **Holdout Test Accuracy:** `{metrics['test_accuracy'] * 100:.2f}%`
- **5-Fold Cross-Validation Mean Accuracy:** `{metrics['cv_accuracy_mean'] * 100:.2f}% (± {metrics['cv_accuracy_std'] * 100:.2f}%)`
- **Macro Average F1-Score:** `{metrics['macro_avg_f1']:.4f}`
- **Weighted Average F1-Score:** `{metrics['weighted_avg_f1']:.4f}`

---

## 2. Detailed Classification Report

```
{report_text}
```

---

## 3. Top Predictive N-Grams (Feature Importance by Class)

"""
    for cls, words in top_features_per_class.items():
        eval_md += f"- **{cls}:** {', '.join(words)}\n"
        
    eval_md += """
---
## 4. Verification and Separation of ML vs Rule Engine
- **Clause Categorization:** Predicted by TF-IDF statistical vectorizer and calibrated linear probability model.
- **Statutory / Student Protection Violations:** Evaluated by separate deterministic compliance engine (`rule_engine.py`) to prevent black-box hallucination.
"""

    with open(MODELS_DIR / "evaluation_report.md", "w") as f:
        f.write(eval_md)
        
    print(f"Model trained successfully!")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Artifacts saved in: {MODELS_DIR}")
    return metrics


if __name__ == "__main__":
    train_and_evaluate_model()
