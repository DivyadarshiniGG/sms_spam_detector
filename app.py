"""
SpamShield AI — Flask Web Application (Fixed Version)
Uses optimal thresholds per model to fix false negatives
"""

from flask import Flask, render_template, request, session, jsonify
import pickle, re, json, os, time
import numpy as np
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from langdetect import detect, LangDetectException
from googletrans import Translator

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

SPAM_WORDS = {
    'free','win','winner','won','prize','cash','claim','offer',
    'urgent','call','text','reply','stop','click','link','verify',
    'account','bank','credit','loan','deal','discount','selected',
    'congratulations','alert','important','limited','exclusive',
    'guaranteed','approved','reward','gift','jackpot','lucky',
    'bonus','earn','money','income','investment','double','percent'
}

app = Flask(__name__)
app.secret_key = 'spamshield_ai_2024_secret'
translator = Translator()

# ── LOAD MODELS ──────────────────────────────────────────────
print("Loading models...", end=' ', flush=True)
with open('models/nb_model.pkl',    'rb') as f: nb         = pickle.load(f)
with open('models/svm_model.pkl',   'rb') as f: svm        = pickle.load(f)
with open('models/lr_model.pkl',    'rb') as f: lr         = pickle.load(f)
with open('models/tfidf_char.pkl',  'rb') as f: tfidf_char = pickle.load(f)
with open('models/tfidf_word.pkl',  'rb') as f: tfidf_word = pickle.load(f)
with open('models/thresholds.pkl',  'rb') as f: thresholds = pickle.load(f)
print("Done!")
print(f"  Thresholds: {thresholds}")

model_map = {'Naive Bayes': nb, 'SVM': svm, 'Logistic Regression': lr}

LANG_NAMES = {
    'en':'English','ta':'Tamil','hi':'Hindi','te':'Telugu',
    'ml':'Malayalam','kn':'Kannada','fr':'French','de':'German','es':'Spanish',
}

# ── HELPERS ──────────────────────────────────────────────────
def extract_meta(text):
    t = str(text)
    return [[
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
    ]]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' urltoken ', text)
    text = re.sub(r'\b\d{10,}\b',   ' phonetoken ', text)
    text = re.sub(r'[^a-z0-9\s]',  ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words or t in SPAM_WORDS]
    return ' '.join(tokens)

def detect_language(text):
    try:    return detect(text)
    except: return 'en'

def translate_to_english(text, src_lang):
    if src_lang == 'en':
        return text, False
    try:
        result = translator.translate(text, dest='en')
        return result.text, True
    except:
        return text, False

def vectorize(text):
    Xc = tfidf_char.transform([text])
    Xw = tfidf_word.transform([text])
    Xm = csr_matrix(np.array(extract_meta(text)))
    return hstack([Xc, Xw, Xm])

def predict_message(message, model_name):
    start = time.time()

    lang_code = detect_language(message)
    lang_name = LANG_NAMES.get(lang_code, lang_code.upper())

    translated, was_translated = translate_to_english(message, lang_code)

    cleaned = clean_text(translated)
    vec     = vectorize(cleaned)

    model   = model_map[model_name]
    proba   = model.predict_proba(vec)[0]
    spam_prob = float(proba[1])

    # ── USE OPTIMAL THRESHOLD (the key fix!) ──────────────────
    threshold = thresholds.get(model_name, 0.35)
    pred = 1 if spam_prob >= threshold else 0

    conf = round(max(spam_prob, 1 - spam_prob) * 100, 1)

    if pred == 1:
        if   conf >= 90: risk = ('Critical Risk',   'critical')
        elif conf >= 70: risk = ('High Risk',        'high')
        else:            risk = ('Medium Risk',      'medium')
    else:
        if   conf >= 85: risk = ('Very Safe',        'safe')
        else:            risk = ('Likely Safe',       'likely_safe')

    xai = explain(cleaned, pred, spam_prob)
    elapsed = round((time.time() - start) * 1000, 1)

    return {
        'label'          : 'SPAM' if pred == 1 else 'HAM',
        'is_spam'        : pred == 1,
        'confidence'     : conf,
        'ham_prob'       : round((1 - spam_prob) * 100, 1),
        'spam_prob'      : round(spam_prob * 100, 1),
        'language_code'  : lang_code,
        'language_name'  : lang_name,
        'was_translated' : was_translated,
        'translated_text': translated if was_translated else None,
        'original_message': message,
        'model_used'     : model_name,
        'threshold_used' : threshold,
        'xai'            : xai,
        'risk_label'     : risk[0],
        'risk_class'     : risk[1],
        'processing_ms'  : elapsed,
    }

def explain(cleaned_text, pred, spam_prob):
    feature_names = (
        list(tfidf_char.get_feature_names_out()) +
        list(tfidf_word.get_feature_names_out()) +
        ['length','digits','uppercase','num_words','has_url',
         'has_spamword','has_currency','exclamations','questions','caps_ratio']
    )
    coef  = lr.coef_[0]
    vec   = vectorize(cleaned_text)
    nz    = vec.nonzero()[1]

    word_scores = []
    for idx in nz:
        if idx < len(coef):
            score = float(vec[0, idx]) * float(coef[idx])
            word_scores.append((feature_names[idx], score))

    word_scores.sort(key=lambda x: x[1], reverse=True)
    spam_triggers = [(w, round(s,4)) for w,s in word_scores if s > 0][:6]
    safe_signals  = [(w, round(abs(s),4)) for w,s in word_scores if s < 0][:4]

    if pred == 1:
        top = [w for w,_ in spam_triggers[:3]]
        summary = (f"Flagged as SPAM (probability {spam_prob*100:.1f}%). "
                   + (f"Key indicators: {', '.join(top)}." if top
                      else "Classified based on overall message pattern."))
    else:
        summary = (f"Classified as HAM (spam probability only {spam_prob*100:.1f}%). "
                   "No significant spam signals found.")

    return {
        'summary'      : summary,
        'spam_triggers': spam_triggers,
        'safe_signals' : safe_signals,
    }

# ── ROUTES ───────────────────────────────────────────────────
@app.route('/')
def index():
    with open('static/results.json') as f:
        results = json.load(f)
    best_model = max(results, key=lambda m: results[m]['F1-Score'])
    best_acc   = results[best_model]['Accuracy']
    return render_template('index.html',
                           best_model=best_model, best_acc=best_acc, results=results)

@app.route('/detect', methods=['GET', 'POST'])
def detect_page():
    result = None; error = None
    if request.method == 'POST':
        message    = request.form.get('message', '').strip()
        model_name = request.form.get('model', 'Logistic Regression')
        if not message:
            error = "Please enter an SMS message."
        elif len(message) > 500:
            error = "Message too long (max 500 characters)."
        else:
            try:
                result = predict_message(message, model_name)
                if 'history' not in session:
                    session['history'] = []
                session['history'].insert(0, {
                    'message'   : message[:70] + ('…' if len(message)>70 else ''),
                    'label'     : result['label'],
                    'confidence': result['confidence'],
                    'model'     : model_name,
                    'language'  : result['language_name'],
                    'risk_class': result['risk_class'],
                })
                session['history'] = session['history'][:25]
                session.modified = True
            except Exception as e:
                error = f"Error: {str(e)}"
    return render_template('detect.html', result=result, error=error)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data       = request.get_json()
    message    = data.get('message','').strip()
    model_name = data.get('model','Logistic Regression')
    if not message:
        return jsonify({'error':'No message'}), 400
    try:
        return jsonify(predict_message(message, model_name))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/performance')
def performance():
    with open('static/results.json') as f:
        results = json.load(f)
    return render_template('performance.html', results=results)

@app.route('/history')
def history():
    hist = session.get('history', [])
    spam_count = sum(1 for h in hist if h['label']=='SPAM')
    return render_template('history.html',
                           history=hist,
                           spam_count=spam_count,
                           ham_count=len(hist)-spam_count)

@app.route('/clear_history')
def clear_history():
    session.pop('history', None)
    return render_template('history.html', history=[], spam_count=0, ham_count=0)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
