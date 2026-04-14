"""
SpamShield AI — FIXED Training Script
Fixes: class imbalance, threshold tuning, better features
"""

import pandas as pd
import numpy as np
import pickle, os, re, json, nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, roc_auc_score,
                             precision_recall_curve)
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

print("=" * 60)
print("  SpamShield AI — Fixed Training Pipeline")
print("=" * 60)

# 1. LOAD DATA
print("\n[1/8] Loading dataset...")
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].copy()
df.columns = ['label', 'message']
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna().drop_duplicates(subset=['message'])
spam_count = df['label'].value_counts()['spam']
ham_count  = df['label'].value_counts()['ham']
print(f"  Ham : {ham_count}  |  Spam: {spam_count}")

# 2. AUGMENT SPAM DATA
print("\n[2/8] Augmenting spam training data...")
extra_spam = [
    "Congratulations you have won a free prize click now",
    "URGENT your account has been selected for a cash reward",
    "Free entry win 1000 pounds text back now",
    "You are a winner claim your prize immediately",
    "Call now to claim your exclusive offer limited time",
    "Winner winner you have been chosen free gift claim",
    "Alert your bank account needs verification click link",
    "Discount offer 80 percent off today only buy now",
    "You won lottery claim prize send details urgent",
    "Free iPhone winner selected click to claim reward",
    "Credit approved apply now low interest rate call",
    "Exclusive deal just for you claim your cash prize",
    "Risk free investment double your money call now",
    "Your number selected win car text back to claim",
    "Urgent response needed account suspended verify now",
    "Aapko inam mila hai abhi claim karo free prize",
    "Winning amount credited click link to withdraw cash",
    "Jackpot winner you get 5000 rupees call now claim",
    "Special offer free data recharge click to activate",
    "Loan approved 50000 instant disbursement call now",
    "OTP do not share bank fraud alert verify identity",
    "Your SIM will be blocked update KYC immediately click",
    "Investment scheme 40 percent monthly returns guaranteed",
    "You have been selected for a free holiday package",
    "Cash prize of 10000 waiting for you call immediately",
    "Win big prizes enter free competition now text WIN",
    "Congratulations ur mobile number won 2000 prize money",
    "Claim ur prize b4 it expires call freephone now",
    "FREE MESSAGE: Congrats u have won a 500 prize reward",
    "U have been selected 2 receivea 350 award call now",
]
extra_ham = [
    "Hey are you coming for lunch today",
    "I will call you back in 5 minutes",
    "Can you send me the notes from class",
    "Meeting postponed to tomorrow same time",
    "Did you finish the assignment",
    "Ok sounds good see you then",
    "What time does the movie start",
    "I am stuck in traffic will be late",
    "Can we reschedule to next week",
    "Happy birthday hope you have a great day",
]
extra_df = pd.DataFrame({
    'label'    : ['spam']*len(extra_spam) + ['ham']*len(extra_ham),
    'message'  : extra_spam + extra_ham,
    'label_num': [1]*len(extra_spam) + [0]*len(extra_ham)
})
df = pd.concat([df, extra_df], ignore_index=True)
print(f"  New spam: {df[df.label=='spam'].shape[0]}  Ham: {df[df.label=='ham'].shape[0]}")

# 3. CLEAN TEXT
print("\n[3/8] Cleaning text...")
stop_words = set(stopwords.words('english'))
SPAM_WORDS = {
    'free','win','winner','won','prize','cash','claim','offer',
    'urgent','call','text','reply','stop','click','link','verify',
    'account','bank','credit','loan','deal','discount','selected',
    'congratulations','alert','important','limited','exclusive',
    'guaranteed','approved','reward','gift','jackpot','lucky',
    'bonus','earn','money','income','investment','double','percent'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' urltoken ', text)
    text = re.sub(r'\b\d{10,}\b',   ' phonetoken ', text)
    text = re.sub(r'[^a-z0-9\s]',  ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words or t in SPAM_WORDS]
    return ' '.join(tokens)

def extract_meta(text):
    t = str(text)
    return [
        len(t),
        sum(c.isdigit()  for c in t),
        sum(c.isupper()  for c in t),
        len(t.split()),
        int(bool(re.search(r'http|www|\.com', t, re.I))),
        int(bool(re.search(r'free|win|prize|cash|claim|urgent|selected|congratulations|reward|bonus|lucky', t, re.I))),
        int(bool(re.search(r'£|\$|€|₹|\d+%', t))),
        t.count('!'),
        t.count('?'),
        sum(c.isupper() for c in t) / max(len(t), 1),
    ]

df['clean'] = df['message'].apply(clean_text)

# 4. VECTORIZE
print("\n[4/8] Building feature matrix...")
tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5),
                              max_features=10000, sublinear_tf=True, min_df=1)
tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                              max_features=8000, sublinear_tf=True, min_df=1)
X_char = tfidf_char.fit_transform(df['clean'])
X_word = tfidf_word.fit_transform(df['clean'])
X_meta = csr_matrix(np.array([extract_meta(m) for m in df['message']], dtype=float))
X = hstack([X_char, X_word, X_meta])
y = df['label_num'].values
print(f"  Feature matrix: {X.shape}")

# 5. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 6. TRAIN MODELS (class_weight='balanced' is the KEY FIX)
print("\n[5/8] Training models with class balancing...")
models = {
    'Naive Bayes': ComplementNB(alpha=0.1),
    'SVM'        : CalibratedClassifierCV(
                       LinearSVC(C=1.0, class_weight='balanced', max_iter=3000), cv=3),
    'Logistic Regression': LogisticRegression(
                       C=2.0, class_weight='balanced', max_iter=2000, solver='lbfgs')
}

results    = {}
cm_data    = {}
thresholds = {}

for name, model in models.items():
    print(f"\n  [{name}]")
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold (maximise F1)
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_proba)
    f1_arr   = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    best_idx = int(np.argmax(f1_arr))
    best_thresh = float(thresh_arr[best_idx]) if best_idx < len(thresh_arr) else 0.3
    best_thresh = float(np.clip(best_thresh, 0.25, 0.55))
    thresholds[name] = round(best_thresh, 3)

    y_pred = (y_proba >= best_thresh).astype(int)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)
    cv   = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

    results[name] = {
        'Accuracy' : round(acc  * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall'   : round(rec  * 100, 2),
        'F1-Score' : round(f1   * 100, 2),
        'ROC-AUC'  : round(auc  * 100, 2),
        'CV-F1-Mean': round(cv.mean() * 100, 2),
        'CV-F1-Std' : round(cv.std()  * 100, 2),
        'Threshold' : best_thresh,
    }
    cm_data[name] = cm.tolist()
    print(f"    Accuracy:{acc*100:.1f}% Recall:{rec*100:.1f}% F1:{f1*100:.1f}% Threshold:{best_thresh:.3f}")

# 7. CHARTS
print("\n[6/8] Saving charts...")
plt.style.use('dark_background')
COLORS = ['#00d4ff', '#ff6b9d', '#c3f73a']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0a0a1a')
for i, (name, cm) in enumerate(zip(results.keys(), cm_data.values())):
    ax = axes[i]
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'],
                ax=ax, linewidths=0.5, annot_kws={'size':14,'weight':'bold'})
    ax.set_title(name, fontsize=13, fontweight='bold', color='white', pad=10)
    ax.set_xlabel('Predicted', color='#aaa'); ax.set_ylabel('Actual', color='#aaa')
    ax.tick_params(colors='#aaa')
plt.suptitle('Confusion Matrices', fontsize=15, color='white', y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig('static/confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

metrics     = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
model_names = list(results.keys())
x, w        = np.arange(len(metrics)), 0.25
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#0a0a1a'); ax.set_facecolor('#0a0a1a')
for i, (mn, col) in enumerate(zip(model_names, COLORS)):
    vals = [results[mn][m] for m in metrics]
    bars = ax.bar(x + i*w, vals, w, label=mn, color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val}', ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
ax.set_ylim([70,104]); ax.set_xticks(x+w)
ax.set_xticklabels(metrics, color='white', fontsize=11)
ax.set_ylabel('Score (%)', color='white'); ax.tick_params(colors='white')
ax.set_title('Model Performance Comparison', fontsize=15, color='white', fontweight='bold', pad=16)
ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white', fontsize=11)
ax.grid(axis='y', color='#333', linestyle='--', alpha=0.5)
for sp in ax.spines.values(): sp.set_edgecolor('#333')
plt.tight_layout()
plt.savefig('static/comparison_chart.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# 8. SAVE
print("\n[7/8] Saving models & results...")
with open('models/nb_model.pkl',   'wb') as f: pickle.dump(models['Naive Bayes'],         f)
with open('models/svm_model.pkl',  'wb') as f: pickle.dump(models['SVM'],                 f)
with open('models/lr_model.pkl',   'wb') as f: pickle.dump(models['Logistic Regression'], f)
with open('models/tfidf_char.pkl', 'wb') as f: pickle.dump(tfidf_char, f)
with open('models/tfidf_word.pkl', 'wb') as f: pickle.dump(tfidf_word, f)
with open('models/thresholds.pkl', 'wb') as f: pickle.dump(thresholds, f)
with open('static/results.json',   'w')  as f: json.dump(results, f, indent=2)
with open('static/cm_data.json',   'w')  as f: json.dump(cm_data,  f, indent=2)

print("\n" + "="*60)
print("  FINAL RESULTS")
print("="*60)
print(f"  {'Model':<22} {'Recall':>7} {'F1':>7} {'Threshold':>10}")
print("  " + "-"*50)
for n, r in results.items():
    print(f"  {n:<22} {r['Recall']:>6}% {r['F1-Score']:>6}% {r['Threshold']:>10.3f}")
print("="*60)
print("\n  Recall = how many SPAM messages we correctly caught")
print("  Higher recall = fewer spam messages missed")
print("\n  Done! Run:  python app.py\n")
