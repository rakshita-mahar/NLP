import os
import csv
import json
import datetime
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
all_patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        tags.append(intent["tag"])

x = vectorizer.fit_transform(all_patterns)
y = tags
clf.fit(x, y)

import pandas as pd

data = {
    "book_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "title": [
        "Harry Potter and the Sorcerer's Stone", 
        "The Hobbit", 
        "To Kill a Mockingbird", 
        "1984", 
        "The Catcher in the Rye", 
        "The Great Gatsby", 
        "The Lord of the Rings", 
        "The Da Vinci Code", 
        "Pride and Prejudice", 
        "The Hunger Games"
    ],
    "author": [
        "J.K. Rowling", 
        "J.R.R. Tolkien", 
        "Harper Lee", 
        "George Orwell", 
        "J.D. Salinger", 
        "F. Scott Fitzgerald", 
        "J.R.R. Tolkien", 
        "Dan Brown", 
        "Jane Austen", 
        "Suzanne Collins"
    ],
    "genre": [
        "Fantasy", 
        "Fantasy", 
        "Fiction", 
        "Dystopian", 
        "Fiction", 
        "Classic", 
        "Fantasy", 
        "Mystery", 
        "Romance", 
        "Dystopian"
    ],
    "description": [
        "A young boy discovers he is a wizard and attends a magical school called Hogwarts.", 
        "A hobbit embarks on a journey to recover treasure guarded by a dragon.", 
        "A young girl grows up in the South during the Great Depression and learns about racial injustice.", 
        "A totalitarian government watches every citizen's move and manipulates the truth.", 
        "A troubled teenager recounts his life experiences in a cynical and detached way.", 
        "A young man becomes infatuated with the mysterious millionaire Gatsby and his ill-fated love story.", 
        "A group of individuals must destroy a powerful ring to save Middle-earth from an evil lord.", 
        "An art historian and a cryptologist solve a series of puzzles leading to a centuries-old secret.", 
        "A spirited young woman navigates issues of love, marriage, and society in Regency-era England.", 
        "In a post-apocalyptic society, young people are forced to participate in a deadly televised competition."
    ]
}

books_df = pd.DataFrame(data)

books_df.to_csv('books.csv', index=False)

print("CSV file 'books.csv' created successfully.")

print(books_df.head())

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_books(book_title, cosine_sim=cosine_sim):
    try:
        idx = books_df[books_df['title'] == book_title].index[0] 
    except IndexError:
        return None 
 
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]  
    book_indices = [i[0] for i in sim_scores]
    
    recommended_books = books_df.iloc[book_indices]
    return recommended_books[['title', 'author', 'genre']]

st.title("Book Recommendation Chatbot")

st.write("""
    Hello! I'm your book recommendation bot. Just type a book title, and I'll suggest similar books based on descriptions.
""")

book_input = st.text_input("Enter a Book Title:")

if book_input:
    recommendations = recommend_books(book_input)
    
    if recommendations is not None:
        st.write(f"Here are some books similar to **'{book_input}'**:")
        for idx, row in recommendations.iterrows():
            st.write(f"**{row['title']}** by {row['author']} (Genre: {row['genre']})")
    else:
        st.write("Sorry, I couldn't find that book in the database. Please try again!")

def preprocess_input(text):
  tokens = word_tokenize(text.lower())
  stop_words = set(stopwords.words("english"))
  tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
  return tokens

def recommend_books(preferences):
  recommendations = []
  for book in books:
    if any(pref in book["genre"].lower() for pref in preferences):
      recommendations.append(book)
    elif any(pref in book["title"].lower() for pref in preferences):
      recommendations.append(book)
  return recommendations

def main():
    st.title("Book Recommendation Chatbot 📚")
    st.write("Tell me what kind of books you like, and I'll recommend something for you!")

    user_input = st.text_input("Enter your preferences (e.g., 'fantasy', 'romance', or a specific title):")
    if st.button("Get Recommendations"):
        if user_input:
            try:
                preferences = preprocess_input(user_input)
                recommendations = recommend_books(preferences)
                if recommendations:
                    st.success("Here are some recommendations:")
                    for book in recommendations:
                        st.write(f"📖 *{book['title']}* by {book['author']} (Genre: {book['genre']})")
                else:
                    st.warning("Sorry, I couldn't find any recommendations based on your preferences.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter your preferences!")
if __name__ == "_main_":
  main()


