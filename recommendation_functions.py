import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Popularity-Based
def popularity_based_recommendations(products,ratings,n=10):
    merged = ratings.merge(products,on="product_id")
    summary = merged.groupby(["product_id","title","category"]).agg(avg_rating=("rating","mean"),review_count=("rating","mean")).reset_index()
    summary = summary.sort_values(["review_count","avg_rating"],ascending=False).head(n)
    return summary

# conntent-Based
def content_based_recommendations(product_id,products,n=5):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(products["description"].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
    indices = pd.Series(products.index,index=products["product_id"])
    idx = indices[product_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key=lambda x:x[1], reverse=True)[1:n+1]
    product_indices = [i[0] for i in sim_scores]
    return products.iloc[product_indices]

# Collaborative filtering
def collaborative_filtering_recommendations(user_id,ratings,products,n=5):
    user_product_matrix = ratings.pivot_table(index="user_id",columns="product_id",values="rating")
    user_mean = user_product_matrix.mean(axis=1)
    matrix_centered = user_product_matrix.sub(user_mean,axis=0)
    sim = matrix_centered.T.corr()
    user_ratings = matrix_centered.loc[user_id].dropna()
    scores = sim[user_ratings.index].dot(user_ratings)
    sim_sums = sim[user_ratings.index].abs().sum()
    pred_ratings = scores/sim_sums
    pred_ratings = pred_ratings.sort_values(ascending=False).head(n)
    recommended = products[products["product_id"].isin(pred_ratings.index)]
    recommended = recommended.copy()
    recommended["avg_rating"]=recommended["product_id"].map(pred_ratings)
    return recommended

# Hybrid
def hybrid_recommendations(user_id,product_id,ratings,products,n=5):
    content_recs = content_based_recommendations(product_id,products,n*2)
    collab_recs = collaborative_filtering_recommendations(user_id,ratings,products,n*2)
    content_ids = set(content_recs["product_id"])
    collab_ids = set(collab_recs["product_id"])
    list(content_ids.intersection(collab_ids))
    if not hybrid_ids:
        return content_recs.head(n)
    hybrid_recs = products[products["product_id"].isin(hybrid_ids)]
    return hybrid_recs.head(n)