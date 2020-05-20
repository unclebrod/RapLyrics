# Initialize the workspace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Always make it pretty.
plt.style.use('ggplot')
sns.set_style(style="whitegrid")

def plot_word_counts(df, artist):
    '''
    Given an artist, this function plots histograms associated with that
    artist's word counts in lyrics.

    Parameters:
    artist (string): the artist of interest
    df (dataframe): dataframe
    '''
    '''
    Given an artist, this function plots histograms associated with that
    artist's raw word counts in lyrics.

    Parameters:
    artist (string): the artist of interest
    df (dataframe): dataframe
    '''
    fig, axs = plt.subplots(2, 1, sharey=True, figsize=(8,12))
    counts = []
    unique_counts = []
    table = str.maketrans("","", string.punctuation.replace("'", ""))
    
    for song in df[df['Artist'] == artist]['Lyrics'].values:
        counts.append(len(song.lower().translate(table).split()))
        unique_counts.append(len(set(song.lower().translate(table).split())))
    
    sns.distplot(counts, ax=axs[0], kde=False,
                 color='mediumslateblue', bins=20)
    axs[0].set_title(f'Total Word Counts in Songs by {artist}')
    axs[0].set_xlabel('Total Words')
    axs[0].set_ylabel('Counts')
    
    sns.distplot(unique_counts, ax=axs[1], kde=False,
                 color='darkslateblue', bins=20)
    axs[1].set_title(f'Unique Word Counts in Songs by {artist}')
    axs[1].set_xlabel('Unique Words')
    axs[1].set_ylabel('Counts')
    plt.show()
    plt.savefig(f'img/word_counts_{artist}.png')

def plot_word_counts_2(df, artist1, artist2):
    '''
    Given a dataframe and two artists, this function plots histograms 
    associated with those artist's raw word counts in lyrics.

    Parameters:
    df (dataframe): dataframe
    artist1 (string): the first artist of interest
    artist2 (string): the second artist of interest
    '''
    fig, axs = plt.subplots(2, 1, sharey=False, figsize=(8,12))
    table = str.maketrans("","", string.punctuation.replace("'", ""))

    counts = []
    unique_counts = []
    for song in df[df['Artist'] == artist1]['Lyrics'].values:
        counts.append(len(song.lower().translate(table).split()))
        unique_counts.append(len(set(song.lower().translate(table).split())))
        
    counts2 = []
    unique_counts2 = []
    for song in df[df['Artist'] == artist2]['Lyrics'].values:
        counts2.append(len(song.lower().translate(table).split()))
        unique_counts2.append(len(set(song.lower().translate(table).split())))
    
    sns.distplot(counts, ax=axs[0], kde=False, norm_hist=True,
                 color='turquoise', bins=20, label=f'{artist1}')
    sns.distplot(counts2, ax=axs[0], kde=False, norm_hist=True,
                 color='orange', bins=20, label=f'{artist2}')
    axs[0].set_title(f'Total Word Count Distribution: {artist1} vs {artist2}')
    axs[0].set_xlabel('Total Words')
    axs[0].set_ylabel('Counts')
    axs[0].legend()
    
    sns.distplot(unique_counts, ax=axs[1], kde=False, norm_hist=True,
                 color='turquoise', bins=20, label=f'{artist1}')
    sns.distplot(unique_counts2, ax=axs[1], kde=False, norm_hist=True,
                 color='orange', bins=20, label=f'{artist2}')
    axs[1].set_title(f'Unique Word Count Distribution:  {artist1} vs {artist2}')
    axs[1].set_xlabel('Unique Words')
    axs[1].set_ylabel('Counts')
    axs[1].legend()
    
    save1 = ''.join(x for x in artist1 if x.isalpha())
    save2 = ''.join(x for x in artist2 if x.isalpha())
    fig.savefig(f'img/word_counts_{save1}_{save2}.png')

def plot_common_words(df, artist):
    '''
    Given an artist, this function plots their 10 most used words, barring
    stop words.

    Parameters:
    artist (string): the artist of interest
    df (dataframe): dataframe
    '''
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    rapwords = {'oh', 'aye', 'yeah', 'yuh', 'huh', 'ha', 'hah', 'ow',
                'ah', 'um', 'umm', '’', 'na', 'nah', 'w/', 'yo', 'w',
                'ooh', 'oooh', 'oohwee', 'oohweee', 'oo', 'ooo', 'oooo',
                'fuck', 'shit', 'bitch', 'damn', 'hell', 'nigga', 'skrt',
                'skrrt', 'skrrrt', "'s", 'wow', 'skrrr', 'fucking', "y'",
                'ya', "'", 'ayy', 'ass', 'niggas', 'uh', 'like'}
    stopwords = stop_words|rapwords

    words = []
    counts = []
    lyrics = " ".join(df[df['Artist'] == artist]['Lyrics'].values)
    cleanlyrics = spacy_tokenizer(lyrics)
    mostcommon = Counter(cleanlyrics).most_common(10)
    for word, count in mostcommon:
        words.append(word.upper())
        counts.append(count)
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x = counts, y = words, palette = 'Blues_r')
    ax.set_title(f'Most Frequently Used Words by {artist}', size=18)
    ax.set_xlabel('Counts', size=14)
    save = ''.join(x for x in artist if x.isalpha())

    fig.savefig(f'img/most_common_{save}.png')

def clean_df(df):
    '''
    Given a dataframe, this function drops rows for songs that are
    snippets, demos, or otherwise incomplete.

    Parameters:
    df (dataframe): dataframe

    Returns:
    df (dataframe): a clean dataframe
    '''
    # Remove songs with < 100 words (typically skits/incomplete songs)
    df.drop(df[df['Word Count'] < 100].index, inplace=True)
        # Remove songs with > 3000 words (typically noise/non-songs)
    df.drop(df[df['Word Count'] > 3000].index, inplace=True)
    # Remove demos
    df.drop(df[(df['Title'].str.contains(r"Demo\)")
                | df['Title'].str.contains(r"\(Demo")
                | df['Title'].str.contains(r"\[Demo"))].index,
            inplace=True)
    # Remove snippets
    df.drop(df[(df['Title'].str.lower().str.contains(r"snippet\)")
                | df['Title'].str.lower().str.contains(r"\(snippet")
                | df['Title'].str.lower().str.contains(r"\[snippet"))]
            .index, inplace=True)
    # Remove unreleased tracks
    df.drop(df[(df['Title'].str.contains(r"Unreleased\)")
                | df['Title'].str.contains(r"\(Unreleased")
                | df['Title'].str.contains(r"\[Unreleased"))].index,
            inplace=True)
    # Remove leaks
    df.drop(df[df['Title'].str.contains(r"\(Leak")
               | df['Title'].str.contains(r"\[Leak")].index, inplace=True)
    # Remove excerpts
    df.drop(df[(df['Title'].str.lower().str.contains(r"excerpt\)")
                | df['Title'].str.lower().str.contains(r"\(excerpt")
                | df['Title'].str.lower().str.contains(r"\[excerpt"))]
            .index, inplace=True)
    # Remove Ted Talks
    df.drop(df[(df['Title'].str.lower().str.contains(r"tedx\)")
                | df['Title'].str.lower().str.contains(r"\(tedx")
                | df['Title'].str.lower().str.contains(r"\[tedx"))].index, inplace=True)
    # Remove Instagram content
    df.drop(df[(df['Title'].str.lower().str.contains("instagram"))]
            .index, inplace=True)
    # Remove Twitter content
    df.drop(df[(df['Title'].str.lower().str.contains("tweets"))].index,
            inplace=True)
    df.drop(df[(df['Title'].str.lower().str.contains("twitter"))].index,
            inplace=True)
    # Remove Reddit content
    df.drop(df[(df['Title'].str.lower().str.contains("reddit"))].index,
            inplace=True)
    # Remove Facebook content
    df.drop(df[(df['Title'].str.lower().str.contains("facebook"))].index,
            inplace=True)
    # Removed edited songs
    df.drop(df[(df['Title'].str.lower().str.contains("edited"))].index,
            inplace=True)
    # Remove Live songs to reduce redundancy
    df.drop(df[(df['Title'].str.lower().str.contains(r"live\)")
                 | df['Title'].str.lower().str.contains(r"\(live")
                 | df['Title'].str.lower().str.contains(r"\[live"))]
            .index, inplace=True)
    # Remove alternate versions
    df.drop(df[(df['Title'].str.contains("alternate"))].index,
            inplace=True)
    # Remove alternate versions
    df.drop(df[(df['Title'].str.contains("version"))].index,
            inplace=True)
    # Remove snippets
    df.drop(df[(df['Lyrics'].str.lower().str
                .contains("lyrics from snippet"))].index, inplace=True)
    df.drop(df[(df['Lyrics'].str.lower().str.contains("compiled from"))]
            .index, inplace=True)
    # Reset the indices
    df.reset_index(drop=True, inplace=True)
    return df

def find_decade(row):
    '''
    Given a row with years, this function creates decades (for apply)
    Parameters:
    row (dataframe): row of a dataframe

    Returns:
    decade (string): the decade associated with the year
    '''
    if (row['Year'] >= 1970) and (row['Year'] < 1990):
        decade = '1970/80s'
    elif (row['Year'] >= 1990) and (row['Year'] < 2000):
        decade = '1990s'
    elif (row['Year'] >= 2000) and (row['Year'] < 2010):
        decade = '2000s'
    elif (row['Year'] >= 2010) and (row['Year'] < 2020):
        decade = '2010s'
    else:
        decade = 0
    return decade

def spacy_tokenizer(song):
    '''
    Given a song, this function tokenizes it, lemmatizes it, and strips it

    Parameters:
    song (string): the lyrics of a song

    Returns:
    tokens (list): a tokenized list of words
    '''
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    rapwords = {'oh', 'aye', 'yeah', 'yuh', 'huh', 'ha', 'hah', 'ow',
                'ah', 'um', 'umm', '’', 'na', 'nah', 'w/', 'yo', 'w',
                'ooh', 'oooh', 'oohwee', 'oohweee', 'oo', 'ooo', 'oooo',
                'fuck', 'shit', 'bitch', 'damn', 'hell', 'nigga', 'skrt',
                'skrrt', 'skrrrt', "'s", 'wow', 'skrrr', 'fucking', "y'",
                'ya', "'", 'ayy', 'ass', 'niggas', 'uh', 'like', 'yo', 'hey',
                'motherfucker', 'pussy', 'ho', 'hoe'}
    stopwords = stop_words|rapwords
    
    doc = nlp(song)
    tokens = ([word.lemma_.lower().strip() for word in doc
               if word.lemma_ != "-PRON-"
               and word.pos_ not in {"PUNCT", "SPACE", "SYM"}
               and word.lemma_ not in stopwords]) 
    return tokens

def create_target(row):
    '''
    This function creates 0/1 targets depending on the desired outcome
    
    Parameters:
    row (dataframe): row of a dataframe

    Returns:
    target (int): 0/1 binary target value
    '''
    if (row['Year'] > 2000):
        target = 1
    else:
        target = 0
    return target

def conf_matrix(actuals, predictions):
    '''
    Given the actual target values and their corresponding predictions,
    this function plots a confusion matrix

    Parameters:
    actuals (list, array): actual target values
    predictions (list, array): predicted target values
    '''
    cf_matrix = confusion_matrix(actuals, predictions)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, 
                annot=labels, 
                fmt='', 
                cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.savefig('img/conf_matrix')