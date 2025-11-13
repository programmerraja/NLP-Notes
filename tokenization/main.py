from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')
nltk.download("averaged_perceptron_tagger_eng")
# nltk.download("wordnet")

# nltk.download(‘punkt’)  # Uncomment this line to download the punkt tokenizer models if not already downloaded
sample_text = "Hello, world! This is a sample text for tokenization."
tokens = word_tokenize(sample_text)

print(tokens)


stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in sample_text.split()]
print(stemmed_tokens)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in sample_text.split()]
print(lemmatized_tokens)


print(pos_tag(tokens))
