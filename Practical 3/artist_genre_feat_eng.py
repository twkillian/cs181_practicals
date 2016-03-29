import numpy as np
import csv
import musicbrainzngs
import pandas as pd

# Adapted from http://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
def remove_non_ascii_chars(word):
    # remove non ascii characters from word and return new word
    return ''.join(letter for letter in word if ord(letter) < 128)


# Read artists csv
artist_df = pd.read_csv('artists.csv')
## set authorization
musicbrainzngs.set_useragent(
    "cs181 practical",
    "0.6",
    "https://github.com/twkillian/cs181_practicals",
)

#initialize genres list
genres = [''] * len(artist_df['artist'].values)
# fetch genre for each artist
i = 0
for artist_id in artist_df['artist'].values:
    # catch error in request
    try:
        result = musicbrainzngs.get_artist_by_id(artist_id,includes=['tags'])
    except musicbrainzngs.WebServiceError as exc:
        print("Something went wrong with the request: %s" % exc)
        # set value to empty string for no genre
        genres[i]=''
    else:
        # parsing the top-voted genre for that artist
        artist = result["artist"]
        if 'tag-list' in artist:
            genre_counts = [(int(tag['count']),tag['name']) for tag in artist['tag-list']]
            sorted_genres = sorted(genre_counts, key=lambda (count,name):  count, reverse=True)
            top_genre = sorted_genres[0][1]
            # store top-voted genre for that artist
            genres[i] = remove_non_ascii_chars(top_genre)

        else:
            genres[i]=''
    print "Finished: {}".format(i)
    i += 1

# Add column of genres to artist_df
artist_df['genre'] = genres
artist_df.to_csv('artists_with_genres.csv')