import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tkinter as tk
from tkinter import scrolledtext

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample FAQ data
faq_data = [
    {"question": "What is your return policy?", "answer": "You can return any product within 30 days of purchase."},
    {"question": "How can I track my order?", "answer": "Use the tracking link sent to your email after shipping."},
    {"question": "Do you offer customer support?", "answer": "Yes, we offer 24/7 customer support via chat and email."},
    {"question": "What payment methods are accepted?", "answer": "We accept credit cards, debit cards, UPI, and net banking."}
]

# Preprocess function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(filtered)

# Prepare TF-IDF
questions = [preprocess(faq['question']) for faq in faq_data]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# Chatbot response function
def chatbot_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    best_match = similarity.argmax()
    score = similarity[0][best_match]
    if score > 0.3:
        return faq_data[best_match]["answer"]
    else:
        return "Sorry, I couldn't find a relevant answer. Please try rephrasing your question."

# GUI with Tkinter
def send_message():
    user_input = entry.get()
    if user_input.strip() == "":
        return
    chat_window.insert(tk.END, "You: " + user_input + "\n")
    response = chatbot_response(user_input)
    chat_window.insert(tk.END, "Bot: " + response + "\n\n")
    entry.delete(0, tk.END)

# Create the main window
root = tk.Tk()
root.title("FAQ Chatbot")

# Chat display
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Arial", 12))
chat_window.pack(padx=10, pady=10)

# Entry field and send button
frame = tk.Frame(root)
entry = tk.Entry(frame, width=50, font=("Arial", 12))
entry.pack(side=tk.LEFT, padx=5)
send_button = tk.Button(frame, text="Send", command=send_message, font=("Arial", 12))
send_button.pack(side=tk.LEFT)
frame.pack(pady=5)

# Run the GUI loop
root.mainloop()
