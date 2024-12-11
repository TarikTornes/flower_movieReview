import os, re
import pandas as pd
import pickle

def load_reviews(data_dir):
    reviews = []
    labels = []
    ids = []
    ratings = []
    dict_pos = {}
    dict_neg = {}

    def parse_filename(file_name):
        match = re.match(r'(\d+)_(\d+)\.txt', file_name)

        if match:
            review_id = int(match.group(1))
            rating = int(match.group(2))
            return review_id, rating

        print("Error: Could not match ID and rating")

        return None, None

    pos_dir = os.path.join(data_dir, 'pos')

    for file_name in os.listdir(pos_dir):

        if file_name.endswith(".txt"):

            review_id, rating = parse_filename(file_name)

            if review_id is not None and rating is not None:

                with open(os.path.join(pos_dir, file_name), 'r') as file:
                    reviews.append(file.read())
                    labels.append(1)
                    ids.append(review_id)
                    ratings.append(rating)
                    dict_pos[review_id] = (file.read(), rating)

    neg_dir = os.path.join(data_dir, 'neg')

    for file_name in os.listdir(neg_dir):

        if file_name.endswith(".txt"):

            review_id, rating = parse_filename(file_name)

            if review_id is not None and rating is not None:

                with open(os.path.join(neg_dir, file_name), 'r') as file:
                    reviews.append(file.read())
                    labels.append(0)
                    ids.append(review_id)
                    ratings.append(rating)
                    dict_neg[review_id] = (file.read(), rating)


    return reviews, labels, ids, ratings, dict_pos, dict_neg







class ReviewLoader:

    def __init__(self, path):
        self.path = path
        reviews, labels, ids, ratings, dict_pos, dict_neg = load_files(path)
        self.reviews = reviews
        self.labels = labels
        self.ids = ids
        self.ratings = ratings
        self.dict_pos = dict_pos
        self.dict_neg = dict_neg


    def get_df(self):
        df = pd.DataFrame({
            'Review ID': self.ids,
            'Review Text': self.reviews,
            'Rating': self.ratings,
            'Sentiment': self.labels
        })
        return df

    def get_reviews(self):
        return self.reviews

if __name__=="__main__":
    def pickle_dataframes(dic, file):
        with open(file, 'wb') as f:
            pickle.dump(dic, f)
            print(f"Dict 1 pickled to {file}")

    print("Starting")
    rl_train = ReviewLoader("../../data/train/")
    rl_test = ReviewLoader("../../data/test/")
    df_train = rl_train.get_df()
    df_test = rl_test.get_df()

    dict_dfs = {"df_train": df_train, "df_test": df_test}
    
    pickle_dataframes(dict_dfs, "movieRev_dfs")


    print("Ending")
        


