import streamlit as st
import pandas as pd
import numpy as np
from recommendation_functions import *

#load data
@st.cache_data
def load_data():
    products = pd.read_csv("products_new.csv")
    ratings = pd.read_csv("reviews_new.csv")
    return products,ratings

products,ratings = load_data()

#App title
st.title("Product Recommendation System")

#sidebar options
st.sidebar.header("Recommendation Options")
rec_type = st.sidebar.radio("Select Recommendation Type",("Popular Products","Content-Based","Collaborative Filtering","Hybrid"))

# popular products
if rec_type == "Popular Products":
    st.header("Top Popular Products")
    n = st.slider("Number of recommendations",5,20,10)
    recommendations = popularity_based_recommendations(products,ratings,n)
st.dataframe(recommendations[["product_id","title","category","avg_rating","review_count"]])

# Content-Based
elif rec_type == 'Content-Based':
    st.header('Find Similar Products')
    product_list = products['title'].values
    selected_product = st.selectbox('Select a product:', product_list)
    selected_product_id = products[products['title'] == selected_product]['product_id'].values[0]
    n = st.slider('Number of recommendations', 5, 20, 5)
    if st.button('Recommend'):
        recommendations = content_based_recommendations(selected_product_id, products, n)
        st.dataframe(recommendations[['product_id', 'title', 'category', 'description']])

# Collaborative Filtering
elif rec_type == 'Collaborative Filtering':
    st.header('Personalized Recommendations')
    user_id = st.selectbox('Select User ID:', ratings['user_id'].unique())
    n = st.slider('Number of recommendations', 5, 20, 5)
    if st.button('Recommend'):
        recommendations = collaborative_filtering_recommendations(user_id, ratings, products, n)
        st.dataframe(recommendations[['product_id', 'title', 'category', 'avg_rating']])

# Hybrid
elif rec_type == 'Hybrid':
    st.header('Hybrid Recommendations')
    user_id = st.selectbox('Select User ID:', np.insert(ratings['user_id'].unique(), 0, 'new_user'))
    product_list = products['title'].values
    selected_product = st.selectbox('Select a product:', product_list)
    selected_product_id = products[products['title'] == selected_product]['product_id'].values[0]
    n = st.slider('Number of recommendations', 5, 20, 5)
    if st.button('Recommend'):
        if user_id == 'new_user':
            st.warning('Showing popular recommendations for new user')
            recommendations = popularity_based_recommendations(products, ratings, n)
        else:
            recommendations = hybrid_recommendations(user_id, selected_product_id, ratings, products, n)
        st.dataframe(recommendations[['product_id', 'title', 'category', 'avg_rating']])












