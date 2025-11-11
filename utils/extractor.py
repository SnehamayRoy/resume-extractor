import pdfplumber, re, os
from pathlib import Path
from docx import Document
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rapidfuzz import process, fuzz

# -------------------------------------------------
# 0. INITIAL SETUP
# -------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------------------------
# 1. LOAD stack.csv + BUILD SETS
# -------------------------------------------------
df = pd.read_csv("stack.csv")

def build_stack_sets():
    lang_set = set(df[df["stack"] == "Language"]["tech"])
    webfw_set = set(df[df["stack"] == "Web"]["tech"])
    plat_set = set(df[df["stack"] == "Cloud"]["tech"])
    db_set = set(df[df["stack"] == "Database"]["tech"])
    return lang_set, webfw_set, plat_set, db_set

LANG_SET, WEBFW_SET, PLAT_SET, DB_SET = build_stack_sets()

# -------------------------------------------------
# 2. BUILD stack_map
# -------------------------------------------------
stack_map = dict(zip(df["tech"], df["stack"]))

# Optional overrides (add any new stack categories you like)
OVERRIDES = {
    "C++": "Systems", "Rust": "Systems", "Go": "Systems",
    "Unity": "GameDev", "Unreal Engine": "GameDev",
    "TensorFlow": "AI/ML", "PyTorch": "AI/ML",
    "Scikit-learn": "AI/ML", "NumPy": "AI/ML", "Pandas": "AI/ML"
}
stack_map.update(OVERRIDES)

# -------------------------------------------------
# 3. TOKEN TO STACK + ALIASES
# -------------------------------------------------
token_to_stack = {tech.lower(): tech for tech in stack_map.keys()}

def add_dynamic_aliases():
    extra = {}
    short_map = {
        'node':'Node.js','nodejs':'Node.js','react':'React.js','reactjs':'React.js',
        'mongo':'MongoDB','express':'Express','bootstrap':'Bootstrap','git':'Git',
        'postman':'Postman','vscode':'VS Code','sublime':'Sublime Text 4',
        'bash':'Bash','cplusplus':'C++','c++':'C++','python':'Python',
        'javascript':'JavaScript','html':'HTML','css':'CSS'
    }
    for k, v in short_map.items():
        extra[k] = v
        extra[k.lower()] = v
    for tech in stack_map.keys():
        lower = tech.lower()
        stripped = re.sub(r'\.(js|net|io|sql)$', '', lower)
        if stripped != lower and stripped not in extra:
            extra[stripped] = tech
    token_to_stack.update(extra)

add_dynamic_aliases()

# -------------------------------------------------
# 4. STOPWORDS + BLACKLIST
# -------------------------------------------------
BLACKLIST = {
    'go','make','using','with','and','the','in','of','to','a','an','i','my','me','at','on',
    'for','by','year','years','experience','good','strong','excellent','proficient',
    'familiar','knowledge','working','team','project','development','software','etc'
}
STOP = set(stopwords.words('english')) | BLACKLIST

# -------------------------------------------------
# 5. PDF/DOCX READER
# -------------------------------------------------
def read_resume(path):
    p = Path(path)
    txt = ""
    if p.suffix.lower() == ".pdf":
        with pdfplumber.open(p) as pdf:
            for page in pdf.pages:
                pg = page.extract_text()
                if pg: txt += pg + "\n"
    elif p.suffix.lower() == ".docx":
        doc = Document(p)
        txt = "\n".join(par.text for par in doc.paragraphs)
    else:
        raise ValueError("Only .pdf/.docx allowed")
    return txt

# -------------------------------------------------
# 6. N-GRAM GENERATION
# -------------------------------------------------
def candidate_ngrams(text, max_n=3):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if re.match(r'^[a-z0-9#.+_-]+$', t)]
    cand = set(tokens)
    for n in range(2, max_n + 1):
        for i in range(len(tokens) - n + 1):
            cand.add(" ".join(tokens[i:i+n]))
    return cand

# -------------------------------------------------
# 7. FUZZY MATCHING
# -------------------------------------------------
def fuzzy_match(cand, threshold=85):
    best, score, _ = process.extractOne(cand, token_to_stack.keys(), scorer=fuzz.token_set_ratio)
    return best if score >= threshold else None

# -------------------------------------------------
# 8. MAIN EXTRACT FUNCTION
# -------------------------------------------------
def extract_skills_from_resume(filepath, threshold=85):
    raw = read_resume(filepath)
    cands = candidate_ngrams(raw, max_n=3)
    resume_tokens = set()
    for line in raw.lower().splitlines():
        line = re.sub(r'[^a-z0-9#.+_-]', ' ', line)
        resume_tokens.update(re.findall(r'[a-z0-9#.+_-]+', line))
    detected = set()
    for c in cands:
        if c in STOP or len(c) < 2:
            continue
        canon = fuzzy_match(c, threshold)
        if canon and canon in resume_tokens:
            detected.add(token_to_stack[canon])
    langs = sorted([t for t in detected if t in LANG_SET])
    webfw = sorted([t for t in detected if t in WEBFW_SET])
    plat = sorted([t for t in detected if t in PLAT_SET])
    db = sorted([t for t in detected if t in DB_SET])
    others = sorted([t for t in detected if t not in (LANG_SET | WEBFW_SET | PLAT_SET | DB_SET)])
    return {
        'languages': langs,
        'web_frameworks': webfw,
        'platforms': plat,
        'databases': db,
        'others': others,
        'all_skills': sorted(detected),
        'total_unique': len(detected)
    }
