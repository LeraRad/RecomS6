import pandas as pd

genome_scores = pd.read_csv('data/raw/genome_scores.csv')
genome_tags = pd.read_csv('data/raw/genome_tags.csv')

# Most informative tags — high average relevance across movies that have them
tag_stats = genome_scores[genome_scores['relevance'] >= 0.5].groupby('tagId').agg(
    avg_relevance=('relevance', 'mean'),
    movie_count=('movieId', 'nunique')
).reset_index()

tag_stats = tag_stats.merge(genome_tags, on='tagId')
tag_stats = tag_stats.sort_values(['movie_count', 'avg_relevance'], ascending=False)
print(tag_stats.head(30))

# Find specific genre/vibe tags
target_tags = ['action', 'comedy', 'thriller', 'horror', 'sci-fi', 'romance', 
               'mystery', 'adventure', 'animation', 'documentary',
               'dark', 'funny', 'suspense', 'emotional', 'thought-provoking',
               'atmospheric', 'plot twist', 'based on a book', 'true story',
               'feel-good', 'dystopia', 'superhero', 'historical']

found = genome_tags[genome_tags['tag'].str.lower().isin(target_tags)]
found_scores = genome_scores[genome_scores['tagId'].isin(found['tagId'])].groupby('tagId').agg(
    avg_relevance=('relevance', 'mean'),
    movie_count=('movieId', 'nunique')
).reset_index()
found_scores = found_scores.merge(genome_tags, on='tagId')
print(found_scores.sort_values('movie_count', ascending=False))