import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize stopwords and lemmatizer
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_duplicate_words(text):
    words = text.split()
    seen = set()
    seen_add = seen.add
    words_no_duplicates = [word for word in words if not (word in seen or seen_add(word))]
    return ' '.join(words_no_duplicates)

def clean_text(text, stopwords=STOPWORDS):
    text = text.lower()
    pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub('', text)
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text)
    text = re.sub(" +", " ", text)
    text = re.sub("\n", " ", text)
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    text = remove_duplicate_words(text)
    return text

def preprocess_for_bm25(text):
    from nltk.tokenize import word_tokenize
    return word_tokenize(clean_text(text))
