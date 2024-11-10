import os
import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

##################################
########## Data Loading ##########
##################################

df = pd.read_csv("../data/books.csv", on_bad_lines='skip')


########################################
########## Data Preprocessing ##########
########################################

# Remove any leading or trailing whitespace in column names
df.columns = df.columns.str.strip()

# Feature Selection
df['features'] = df['title'] + ' ' + df['authors'] + ' ' + df['publisher']

# Get matrix vectors
def preprocessed_data(data):
    data['features'] = data['features'].str.lower()
    data['features'] = data['features'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    data['features'] = data['features'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['features'])
    
    return tfidf_matrix, tfidf

tfidf_matrix, tfidf = preprocessed_data(df)


###################################
########## Data Modeling ##########
###################################


def recommender_engine(query, top_n=5):
    query_vec = tfidf.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix)
    similar_books_indices = cosine_similarities.argsort()[0][-top_n:][::-1]
    top_books = df.iloc[similar_books_indices][['title', 'authors', 'publication_date']]
    
    return top_books


#########################################
########## Streamlit Interface ##########
#########################################


def main():
    
    # Streamlit app layout
    st.title("Book Recommendation System")

    # User input for the book query
    query = st.text_input("Enter book title or description:")

    # Dynamic slider for number of recommendations
    top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)

    # Display recommendations if query is provided
    if query:
        recommendations = recommender_engine(query, top_n)
        st.write("Recommendations:")
        for _, row in recommendations.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"{row['authors']}")
            st.write(f"{row['publication_date']}")
            st.write("---")  # Add a separator between books
        # st.write(recommendations)
        

if __name__ == "__main__":
    main()