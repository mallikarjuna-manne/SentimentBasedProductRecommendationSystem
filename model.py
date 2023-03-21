import numpy as np
import pandas as pd
import pickle


class SentimentBasedRecommenderSystem:
    best_snmt_model = "logistic_reg_final_model.pkl"
    tfidf_vect = "tfidf_vectorizer.pkl"
    best_recomm = "best_recommender_model.pkl"
    sentiment_df = "sentiment_dataframe.pkl"

    def __init__(self):
        # Load all pickle files
        self.sentiment_model = pickle.load(open(SentimentBasedRecommenderSystem.best_snmt_model, 'rb'))
        self.tfidf_vectorizer = pd.read_pickle(SentimentBasedRecommenderSystem.tfidf_vect)
        self.user_final_rating = pickle.load(open(SentimentBasedRecommenderSystem.best_recomm, 'rb'))
        self.snmt_data = pickle.load(open(SentimentBasedRecommenderSystem.sentiment_df, 'rb'))

    def top_5_recommendations(self, UserName):
        if UserName not in self.user_final_rating.index:
            print(f"The User '{UserName}' doesn't exist. Please provide a valid user name")
            return None
        else:
            # Get top 20 recommended products from the best recommendation model
            top_20_recommended_products = list(
                self.user_final_rating.loc[UserName].sort_values(ascending=False)[0:20].index)
            # Get only the recommended products from the prepared dataframe "df_sent_Ana"
            df_top_20_products = self.snmt_data[self.snmt_data.id.isin(top_20_recommended_products)]
            # For these 20 products, get their user reviews and pass them through TF-IDF vectorizer to convert the data into suitable format for modeling
            X = self.tfidf_vectorizer.transform(df_top_20_products["lemmatized_reviews"].values.astype(str))
            # Use the best sentiment model to predict the sentiment for these user reviews
            df_top_20_products['predicted_snmt'] = self.sentiment_model.predict(X)
            # Create a new dataframe "predicted_df" to store the count of positive user sentiments
            predicted_df = pd.DataFrame(df_top_20_products.groupby(by='name').sum()['predicted_snmt'])
            predicted_df.columns = ['pos_snmt_count']
            # Create a column to measure the total sentiment count
            predicted_df['total_snmt_count'] = df_top_20_products.groupby(by='name')['predicted_snmt'].count()
            # Create a column that measures the % of positive user sentiment for each product review
            predicted_df['positive_snmt_per'] = np.round(predicted_df['pos_snmt_count'] / predicted_df['total_snmt_count'] * 100, 2)
            # Return top 5 recommended products to the user
            result = list(predicted_df.sort_values(by='positive_snmt_per', ascending=False)[:5].index)
            return result
