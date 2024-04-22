import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class PostRecommendationModel:
    def __init__(self, post_data_file, user_data_file, view_data_file):
        self.post_data_file = post_data_file
        self.user_data_file = user_data_file
        self.view_data_file = view_data_file
        self.post_data = None
        self.user_data = None
        self.view_data = None
        self.rating_popular_post_user = None
        self.rating_popular_post_title = None
        self.model_user = None
        self.model_title = None
        self.load_data()
        self.preprocess_data()
        self.build_models()

    def load_data(self):
        self.post_data = pd.read_csv(self.post_data_file)
        self.user_data = pd.read_csv(self.user_data_file)
        self.view_data = pd.read_csv(self.view_data_file)

    def preprocess_data(self):
        self.post_data["Valuable"] = np.random.randint(1, 6, len(self.post_data))
        combined_data = pd.merge(self.view_data, self.post_data, on='post_id')
        post_rating_count = combined_data.dropna(axis=0, subset=['title']).groupby(by=['title'])['Valuable'].count().reset_index().rename(columns={'Valuable': 'totalValuableCount'})[['title', 'totalValuableCount']]
        rating_with_totalValuableCount = combined_data.merge(post_rating_count, on='title', how='left')
        popularity_threshold = 13
        self.rating_popular_post = rating_with_totalValuableCount.query('totalValuableCount >= @popularity_threshold')

    def build_models(self):
        self.rating_popular_post_user = self.rating_popular_post.drop_duplicates(['user_id', 'title']).pivot(index='user_id', columns='title', values='Valuable').fillna(0)
        self.rating_popular_post_title = self.rating_popular_post.drop_duplicates(['user_id', 'title']).pivot(index='title', columns='user_id', values='Valuable').fillna(0)
        self.model_user = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_user.fit(self.rating_popular_post_user.values)
        self.model_title = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_title.fit(self.rating_popular_post_title.values)

    def recommend_posts_for_user(self, user_id):
        query_index = self.rating_popular_post_user.index.get_loc(user_id)
        distances, indices = self.model_user.kneighbors(self.rating_popular_post_user.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)
        recommended_posts = [self.rating_popular_post_user.columns[i] for i in indices.flatten()]
        return recommended_posts[1:]  # Exclude the first as it's the input user's own post

    def recommend_similar_posts(self, post_title):
        query_index = self.rating_popular_post_title.index.get_loc(post_title)
        distances, indices = self.model_title.kneighbors(self.rating_popular_post_title.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)
        similar_posts = [self.rating_popular_post_title.index[i] for i in indices.flatten()]
        return similar_posts[1:]  # Exclude the first as it's the input post itself
