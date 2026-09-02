### ACTION

import numpy as np
import pandas as pd
from utils import *
import os

def load_dataframes_ratings():
    ds_ratings_path = get_dataset("arashnic/book-recommendation-dataset")

    books_df =pd.read_csv(os.path.join(ds_ratings_path, "Books.csv"),
                        dtype={"Year-Of-Publication": "string"})
    users_df =pd.read_csv(os.path.join(ds_ratings_path, "Users.csv"))
    ratings_df =pd.read_csv(os.path.join(ds_ratings_path, "Ratings.csv"))

    ratings_df["Book-Rating"] = ratings_df["Book-Rating"].astype(float)

    # rename fields
    books_df.rename(columns={'Book-Title': 'title'}, inplace=True)
    books_df.rename(columns={'Book-Author': 'author'}, inplace=True)
    books_df.rename(columns={'Year-Of-Publication': 'pub_year'}, inplace=True)

    # remove implicit 0 ratings
    ratings_df = ratings_df[ratings_df["Book-Rating"] != 0]

    return books_df, ratings_df, users_df

### ACTION

def merge_by_title(books_df, ratings_df):
    books_df = books_df.copy()
    ratings_df = ratings_df.copy()

    # use clean titles
    books_df['title_clean'] = books_df['title'].str.replace(r'\s*\([^)]*\)\s*$', '', regex=True)
    books_df['title'] = books_df['title_clean']
    books_df = books_df.drop(columns='title_clean')

    # use smallest ISBN when mapping title and author -> ISBN
    isbn_map = books_df.groupby(['title', 'author'], dropna=False)['ISBN'].min().reset_index()
    isbn_map = isbn_map.rename(columns={'ISBN': 'ISBN_merged'})

    isbn_to_merged = books_df.merge(isbn_map, on=['title', 'author'], how='left').set_index('ISBN')['ISBN_merged'].to_dict()

    books_df = books_df.merge(isbn_map, on=['title', 'author'], how='left')
    books_df = books_df.drop_duplicates(subset='ISBN_merged', keep='first')
    books_df['ISBN'] = books_df['ISBN_merged']
    books_df = books_df.drop(columns='ISBN_merged')

    ratings_df['ISBN'] = ratings_df['ISBN'].map(isbn_to_merged).fillna(ratings_df['ISBN'])
    ratings_df = ratings_df.groupby(['User-ID', 'ISBN'], as_index=False)['Book-Rating'].mean()

    return books_df, ratings_df

### ACTION

def filter_rankings_freq(df):
    """Filters out rows, to leave keep min ratings per book and min ratings per user"""
    df = df.copy()

    # filter out books and users with few ratings
    MIN_RATINGS_USER = 3
    MIN_RATINGS_BOOK = 1
    book_counts = df["ISBN"].value_counts()
    user_counts = df["User-ID"].value_counts()

    i = 0
    while True:
        i += 1
        old_size = len(df)

        user_counts = df["User-ID"].value_counts()
        df = df[df["User-ID"].isin(user_counts[user_counts >= MIN_RATINGS_USER].index)]

        book_counts = df["ISBN"].value_counts()
        df = df[df["ISBN"].isin(book_counts[book_counts >= MIN_RATINGS_BOOK].index)]

        if len(df) == old_size:
            break

    print(f"Iterations: {i}")

    return df

### ACTION

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

class RecomModelCF:
    def __init__(self, books_df, k, raw=False):
        self.books_df = books_df.copy()
        self._k = k
        self._raw = raw

    def train(self, raitings_train_df):    
        user_c = raitings_train_df["User-ID"].astype("category") # user_id -> user_index
        book_c = raitings_train_df["ISBN"].astype("category") # book_isbn -> book_index
        # build a sparse matrix -> book rating at position (user_index, book_index)
        self.matrix = csr_matrix(
            (raitings_train_df["Book-Rating"], (user_c.cat.codes, book_c.cat.codes))
        )

        self.user_ids = user_c.cat.categories
        self.book_isbns = book_c.cat.categories

        if self._raw:
            return
        
        self.U, self.sigma, self.Vt = svds(self.matrix, k=self._k)

    def recommend(self, isbn, n=10):
        if self._raw:
            return self._recommend_raw(isbn, n)
        
        book_vectors = self.Vt.T

        idx = self.book_isbns.get_loc(isbn)

        sims = cosine_similarity(
            book_vectors[idx].reshape(1, -1),
            book_vectors
        )[0]

        return self._get_books(sims, n)

    def has_book(self, isbn):
        return isbn in self.book_isbns

    def _get_books(self, sims, n):
        top_idx = sims.argsort()[::-1][1:n+1]
        top_isbns = self.book_isbns[top_idx]

        result = self.books_df[self.books_df['ISBN'].isin(top_isbns)].copy()
        result['score'] = result['ISBN'].map(dict(zip(top_isbns, sims[top_idx])))
        result = result.sort_values('score', ascending=False)

        return result

    def _recommend_raw(self, isbn, n=10):
        idx = self.book_isbns.get_loc(isbn)

        sims = cosine_similarity(
            self.matrix[:, idx].T,
            self.matrix.T
        )[0]

        return self._get_books(sims, n)

### ACTION

def load_dataframes_CB():
    ds_info_path = get_dataset("abhijitdalal26/books-dataset-with-description-85k-books")

    books_info_df = pd.read_csv(os.path.join(ds_info_path, "books_dataset.csv"))
    print(books_info_df.shape)
    print(books_info_df.columns)

    # convert to ISBN10, drop if cannot convert
    import isbnlib
    def to_isbn10(isbn13):
        isbn13 = str(isbn13)
        if isbnlib.is_isbn13(isbn13):
            return isbnlib.to_isbn10(isbn13) or None
        return None

    books_info_df['ISBN'] = books_info_df['isbn13'].apply(to_isbn10)
    books_info_df = books_info_df.dropna(subset=['ISBN'])
    books_info_df = books_info_df.drop(columns=['isbn13'])
    books_info_df.insert(0, 'ISBN', books_info_df.pop('ISBN'))

    # fill missing ratings count with 0
    books_info_df['ratings_count'] = books_info_df['ratings_count'].fillna(0).clip(lower=0).astype(int)
    # fill missing ratings with mean rating
    avg_rating = books_info_df['avg_rating'].mean()
    books_info_df['avg_rating'] = books_info_df['avg_rating'].fillna(avg_rating).clip(lower=0.0, upper=5.0).astype(float)

    return books_info_df


### ACTION

import ast
def prep_genres_CB(genres):
    """ Formats all genres into list[str]. Splits genres like "history, fiction" into "history" and "fiction". """
    if isinstance(genres, list):
        genre_list = genres
    elif type(genres) is not str or genres.strip() == "":
        return []
    else:
        try:
            genre_list = ast.literal_eval(genres)
            if not isinstance(genre_list, list):
                genre_list = [str(genre_list)]
        except (ValueError, SyntaxError):
            genre_list = [genres]

    result = set()
    for g in genre_list:
        result.update(part.strip().lower() for part in g.split(','))
    return list(result)


### ACTION

def merge_by_title_CB(df):
    df = df.copy()

    counts = df.groupby(['title', 'author'])['ISBN'].nunique()
    duplicates = counts[counts > 1]
    print(f"Total books: {df.shape[0]} books with duplicates: {len(duplicates)}")

    print("Books before: ", df.shape)

    # isbn to other same book isbns, keep smallest isbn
    isbn_map = df.groupby(['title', 'author'], dropna=False)['ISBN'].min().reset_index()
    isbn_map = isbn_map.rename(columns={'ISBN': 'ISBN_merged'})

    # merged genres
    genre_union = df.groupby(['title', 'author'], dropna=False)['genres'].agg(
        lambda lists: list(set().union(*lists))
    ).reset_index()
    genre_union = genre_union.rename(columns={'genres': 'genres_merged'})

    # merged descriptions
    desc_union = df.groupby(['title', 'author'], dropna=False)['description'].agg(
        lambda descs: [d for d in descs if isinstance(d, str)]
    ).reset_index()
    desc_union = desc_union.rename(columns={'description': 'description_merged'})

    # merged series_name (prefer non-null value if any book in the group has it)
    series_fill = df.groupby(['title', 'author'], dropna=False)['series_name'].agg(
        lambda vals: next((v for v in vals if pd.notna(v)), np.nan)
    ).reset_index()
    series_fill = series_fill.rename(columns={'series_name': 'series_name_filled'})

    # merge
    df = df.merge(isbn_map, on=['title', 'author'], how='left')
    df = df.merge(genre_union, on=['title', 'author'], how='left')
    df['genres'] = df['genres_merged']

    df = df.merge(desc_union, on=['title', 'author'], how='left')
    df['description'] = df['description_merged']

    df = df.merge(series_fill, on=['title', 'author'], how='left')
    df['series_name'] = df['series_name_filled']

    df = df.drop_duplicates(subset='ISBN_merged', keep='first')

    df['ISBN'] = df['ISBN_merged']

    df = df.drop(columns=['ISBN_merged', 'genres_merged', 'description_merged', 'series_name_filled'])

    return df

### ACTION

def assing_quality_score_CB(df):
    df = df.copy()

    avg_rating = df['avg_rating'].mean()
    median_ratings_count = df['ratings_count'].median()

    def quality_score(row):
        v = row['ratings_count']
        R = row['avg_rating']
        m = median_ratings_count
        C = avg_rating

        return (v / (v + m)) * R + (m / (v + m)) * C

    df['quality_score'] = df.apply(quality_score, axis=1)

    return df

### ACTION

from collections import Counter

def filter_genre_freq_CB(df):
    """ Filter in loop to keep min genre freq and min genres count per book. """
    df = df.copy()

    MIN_GENRE_FREQ = 4
    MIN_GENRE_PER_BOOK = 2

    print(f"Books before: {df.shape[0]}")

    i = 0
    while True:
        i += 1

        genre_counts = Counter()
        df['genres'].apply(genre_counts.update)

        if i == 1:
            print(f"Unique genres before: {len(genre_counts)}")

        # filter genres by min frequency
        genres_to_keep = Counter({g: c for g, c in genre_counts.items() if c > MIN_GENRE_FREQ})
        genres_to_keep = set(genres_to_keep.keys())

        # Remove unused genres from books
        df['genres'] = df['genres'].apply(lambda genres: [x for x in genres if x in genres_to_keep])

        len_before = df.shape[0]

        # Filter books by min genre count
        df = df[df['genres'].apply(len) >= MIN_GENRE_PER_BOOK]

        # break if no change
        if df.shape[0] == len_before:
            break

    print(f"Unique genres after: {len(genre_counts)}")
    print(f"Iterations: {i}")
    print(f"Books after: {df.shape[0]}")
    return df


### ACTION

import numpy as np
from sklearn.decomposition import TruncatedSVD

class RecomModelGenreBased:
    def __init__(self, df, embedding_dim=50):
        df = df.copy()

        genres = set()
        df['genres'].apply(genres.update)
        self.genre_to_idx = {g: i for i, g in enumerate(genres)}

        df['genres_onehot'] = df['genres'].apply(self.genres_to_onehot)

        self.df = df.reset_index(drop=True)

        # one hot matrix
        onehot_matrix = np.stack(self.df['genres_onehot'].values)

        # IDF
        doc_freq = onehot_matrix.sum(axis=0) # how many books have each genre
        self.idf = np.log(len(self.df) / (doc_freq + 1)) # inverse document frequency
        weighted_matrix = onehot_matrix * self.idf

        # genre-genre co-occurrence matrix
        cooc = weighted_matrix.T @ weighted_matrix # (n_genres, n_genres)
        np.fill_diagonal(cooc, 0)

        # reduce to dense genre embeddings via SVD
        svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        genre_embeddings = svd.fit_transform(cooc)  # (n_genres, embedding_dim)

        # book vector = sum of its genres' embeddings
        self.matrix = onehot_matrix @ genre_embeddings  # (n_books, embedding_dim)

        self.isbn_to_row = {isbn: i for i, isbn in enumerate(self.df['ISBN'])}

        # precompute norms
        self.norms = np.linalg.norm(self.matrix, axis=1)

    def print_info(self):
        # print weights of most common and rarest genres
        weights = dict(zip(self.genre_to_idx.keys(), self.idf))
        sorted_weights = sorted(weights.items(), key=lambda x: x[1])
        print("Lowest (most common):", sorted_weights[:4])
        print("Highest (rarest):", sorted_weights[-4:])

    def genres_to_onehot(self, genres):
        vec = np.zeros(len(self.genre_to_idx))
        for g in genres:
            vec[self.genre_to_idx[g]] = 1
        return vec

    def has_book(self, isbn):
        return isbn in self.isbn_to_row

    def recommend(self, isbn, exclude_same_series=True, exclude_same_author=False, top_n=5):
        row = self.isbn_to_row[isbn]
        query_vec = self.matrix[row]

        
        query_norm = np.linalg.norm(query_vec)
        sims = self.matrix @ query_vec / (self.norms * query_norm + 1e-9)

        result = self.df[['ISBN', 'title', 'author', 'quality_score', 'series_name']].assign(similarity=sims)
        result = result.drop(row)

        if exclude_same_series:
            query_series = self.df.loc[row, 'series_name']
            if pd.notna(query_series):
                result = result[result['series_name'] != query_series]

        if exclude_same_author:
            query_author = self.df.loc[row, 'author']
            if pd.notna(query_author):
                result = result[result['author'] != query_author]

        result = result.sort_values(['similarity', 'quality_score'], ascending=False)
        return result.head(top_n)


### ACTION

def prep_descriptions_CB(df):
    MIN_LEN = 16
    MAX_LEN = 384
    df = df.copy()

    print(f"Before: {df.shape[0]}")

    # keep only descriptions with length within a limit
    df['description'] = df['description'].apply(lambda x: [y for y in x if len(y.split()) >= MIN_LEN and len(y.split()) < MAX_LEN])

    # Keep only books with atleast one description
    df = df[df['description'].apply(len) > 0]

    # Keep only the first description, NOTE: the embedding could be averaged out from multiple descriptions if needed
    df['description'] = df['description'].apply(lambda x: str(x[0]))

    print(f"After: {df.shape[0]}")

    return df


### ACTION

from sentence_transformers import SentenceTransformer

class RecomModelDescBased():
    def __init__(self, df):
        self.df = df.copy()
        self.create_embeddings(self.df, overwrite=False)
        self.norms = np.linalg.norm(self.desc_embeddings, axis=1)

    def create_embeddings(self, df, overwrite):
        df = df.reset_index(drop=True)
        self.isbn_to_row = {isbn: i for i, isbn in enumerate(df['ISBN'])}
    
        if os.path.exists('description_embeddings.npy') and not overwrite:
            print("Using cached embeddings.")
            self.desc_embeddings = np.load('description_embeddings.npy')
            return
        
        self.embedder_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        self.desc_embeddings = self.embedder_model.encode(df['description'].tolist(), show_progress_bar=True)

        np.save('description_embeddings.npy', self.desc_embeddings)

    def has_book(self, isbn):
        return isbn in self.isbn_to_row

    def recommend(self, isbn, exclude_same_series=True, exclude_same_author=False, top_n=5):
        row = self.isbn_to_row[isbn]
        query_vec = self.desc_embeddings[row]

        query_norm = np.linalg.norm(query_vec)
        sims = self.desc_embeddings @ query_vec / (self.norms * query_norm + 1e-9)

        result = self.df[['ISBN', 'title', 'author', 'quality_score', 'series_name']].assign(similarity=sims)
        result = result.drop(row)

        if exclude_same_series:
            query_series = self.df.loc[row, 'series_name']
            if pd.notna(query_series):
                result = result[result['series_name'] != query_series]
        
        if exclude_same_author:
            query_author = self.df.loc[row, 'author']
            if pd.notna(query_author):
                result = result[result['author'] != query_author]

        result = result.sort_values(['similarity', 'quality_score'], ascending=False)
        return result.head(top_n)

from rapidfuzz import process, fuzz

class CombinedRecommender():
    def __init__(self, books_df, ratings_df, books_genre_df, books_desc_df):
        self.books_df = books_df.copy()
        self.ratings_df = ratings_df.copy()
        self.books_genre_df = books_genre_df.copy()
        self.books_desc_df = books_desc_df.copy()

        self.model_cf = RecomModelCF(books_df=books_df, k=0, raw=True)
        self.model_cf.train(ratings_df)
        self.model_genre = RecomModelGenreBased(df=books_genre_df, embedding_dim=100)
        self.model_desc = RecomModelDescBased(books_desc_df)

        rated_isbns = ratings_df["ISBN"].unique()
        self.availible_books_by_users = books_df[
            books_df["ISBN"].isin(rated_isbns)
        ][["ISBN", "title", "author"]].reset_index(drop=True)

        merged = pd.merge(
            books_genre_df[['ISBN', 'title', 'author']],
            books_desc_df[['ISBN', 'title', 'author']],
            on='ISBN', how='outer', suffixes=('', '_desc')
        )
        merged['title'] = merged['title'].fillna(merged['title_desc'])
        merged['author'] = merged['author'].fillna(merged['author_desc'])
        self.availible_books_by_content = merged[['ISBN', 'title', 'author']].reset_index(drop=True)

    def find_title(self, title, limit=5):
        def get_results(df):
            results = process.extract(
                title,
                df["title"],
                scorer=fuzz.WRatio,
                limit=limit
            )

            indices = [r[2] for r in results]

            return df.iloc[indices].assign(
                score=[r[1] for r in results]
            ).reset_index(drop=True)

        return {
            'by_users': get_results(self.availible_books_by_users),
            'by_content': get_results(self.availible_books_by_content)}

    def recommend_by_isbn(self, isbn, exclude_same_series = True, n=5):
        return {
            'by_users' : self.model_cf.recommend(isbn, n)[['title', 'author']] if self.model_cf.has_book(isbn) else [],
            'by_genre' : self.model_genre.recommend(isbn=isbn, exclude_same_series=exclude_same_series, top_n=n)[['title', 'author']] if self.model_genre.has_book(isbn) else [],
            'by_description' : self.model_desc.recommend(isbn=isbn, exclude_same_series=exclude_same_series, top_n=n)[['title', 'author']] if self.model_desc.has_book(isbn) else [],
        }

    def recommend_by_title(self, title, exclude_same_series=True, n=5):
        found_titles = self.find_title(title=title, limit=1)

        by_users_row = found_titles['by_users'].iloc[0] if len(found_titles['by_users']) > 0 else None
        by_users_book = by_users_row if by_users_row is not None and by_users_row['score'] >= 70 else None

        by_content_row = found_titles['by_content'].iloc[0] if len(found_titles['by_content']) > 0 else None
        by_content_book = by_content_row if by_content_row is not None and by_content_row['score'] >= 70 else None

        return {
            'by_users': (
                {
                    'found': by_users_book['title'],
                    'recommendations': self.model_cf.recommend(by_users_book['ISBN'], n)[['title', 'author']]
                }
                if by_users_book is not None
                else {'found': None, 'recommendations': []}
            ),
            'by_genre': (
                {
                    'found': by_content_book['title'],
                    'recommendations': self.model_genre.recommend(
                        isbn=by_content_book['ISBN'],
                        exclude_same_series=exclude_same_series,
                        top_n=n
                    )
                }
                if by_content_book is not None and self.model_genre.has_book(by_content_book['ISBN'])
                else {'found': None, 'recommendations': []}
            ),
            'by_desc': (
                {
                    'found': by_content_book['title'],
                    'recommendations': self.model_desc.recommend(
                        isbn=by_content_book['ISBN'],
                        exclude_same_series=exclude_same_series,
                        top_n=n)
                }
                if by_content_book is not None and self.model_desc.has_book(by_content_book['ISBN'])
                else {'found': None, 'recommendations': []}
            )
        }



####
books_df, ratings_df, users_df = load_dataframes_ratings()
books_df, ratings_df = merge_by_title(books_df=books_df, ratings_df=ratings_df)
df_ratings_filtered = filter_rankings_freq(ratings_df)

books_info_df = load_dataframes_CB()
books_info_df['genres'] = books_info_df['genres'].apply(prep_genres_CB)
books_info_df = merge_by_title_CB(books_info_df)
books_info_df = assing_quality_score_CB(books_info_df)
books_genre_df = filter_genre_freq_CB(books_info_df)
books_desc_df = prep_descriptions_CB(books_info_df)


combined_model = CombinedRecommender(books_df=books_df,
                                     ratings_df=df_ratings_filtered,
                                     books_genre_df=books_genre_df,
                                     books_desc_df=books_desc_df)

from pprint import pprint

#pd.set_option('display.max_colwidth', None)
#print(combined_model.find_title("Crime and Punishment", limit=3))
# pprint(combined_model.recommend_by_isbn(isbn='0439554934', exclude_same_series = True, n=3))
pprint(combined_model.recommend_by_title(title="Game of Thrones", n=3))
