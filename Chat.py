import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.chat.util import Chat, reflections

# Ignore warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('popular', quiet=True)

# Load and preprocess the corpus
with open('E:\\ChatBot.txt', 'r', errors='ignore') as f:
    raw = f.read().lower()

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Define greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "yo")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    spar_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        spar_response = spar_response + "I don't understand you"
    else:
        spar_response = spar_response + sent_tokens[idx]
    sent_tokens.remove(user_response)
    return spar_response

# Define pattern-based chatbot pairs
pairs = [
    ['my name is (.*)', ['Hi %1']],
    ['(hey|hello|yo|hi|hola|what’s good?)',
     ['Hello, I’m glad to talk to you!',
      'Nice to meet you!', "Hope you’re doing good", "Hi there! How can I help you?"]],
    ['(.*) is fun', ['%1 is indeed fun']],
    ['(what is your name?|what are you?|who are you?|your name please?)',
     ['Hi I’m Spar', 'Call me Spar', 'I’m Spar, nice to meet you']],
    ['(bye|see you later|goodbye|nice chatting with you)',
     ['Goodbye! Take care!', 'See you later!', 'Have a great day!']],
    ['(thanks|thank you|that’s helpful|awesome, thanks|thanks for helping me)',
     ['Happy to help!', 'Anytime', 'No problem, you’re welcome']],
    ['(how can you help me?|what can you do?|what help do you provide?|help?|how can you be helpful?|what support do you offer?|what do you know?)',
     ['I can answer questions about chatbots and machine learning.']],
    ['(what is a chatbot?|what do you know about chatbots?|tell me about chatbots?|chatbots?)',
     ['A chatbot is a computer program designed to simulate conversation with human users, especially over the Internet.']],
    ['(what is machine learning?|tell me about machine learning?|machine learning?)',
     ['Machine learning is a branch of AI focused on building applications that learn from data and improve their accuracy over time without being programmed to do so.']],
    
]

my_reflections = {"hi": "hello"}
chat = Chat(pairs, my_reflections)

# Main chatbot loop
print("Spar: My name is Spar. I will answer your queries about Chatbots. If you want to exit, type Bye!")

while True:
    user_response = input().lower()
    if user_response == 'bye':
        print("Spar: Goodbye! Take care!")
        break
    elif user_response in ('thanks', 'thank you'):
        print("Spar: You are welcome.")
        break
    elif greeting(user_response) is not None:
        print(f"Spar: {greeting(user_response)}")
    else:
        
        print(f"Spar: {response(user_response)}")

