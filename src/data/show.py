import pandas as pd

def main():
    m_cols = ['movie_id', 'title', 'genre']
    movies = pd.read_csv('./ml-10M100K/movies.dat', names=m_cols, sep='::', encoding='latin-1', engine='python')
    movies['genre'] = movies.genre.apply(lambda x:x.split('|'))
    print( movies.head())


if __name__ == "__main__":
    main()
