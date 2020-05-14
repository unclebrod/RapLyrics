# Rap Lyrics: An Analysis of Songs Across the Years
The goal of this project is to explore rap lyrics, gathered from 100 rappers from the late 1970s to present day, using natural language processing. I explored both supervised and unsupervised learning, as well as a number of visuals.

# Background & Inspiration
I recently began learning about natural language processing and the many techniques data scientists have developed to make sense of text. A month ago, I completed a project where I scraped data from [Metacritic](https://www.metacritic.com) and explored how various genres had been reviewed historically based on their rating system (if interested, feel free to read about it [here](https://github.com/unclebrod/ReviewsByGenre)). I thought a natural extension of this project would be to look into my favorite genre, rap, and see what insights I could gain from its lyrics. I thought rap would be especially interesting because it's a musicform which relies on its artists producing lyrical content in much higher volumes than other genres, and as such I felt confident that I could learn plenty.

# Goals
My goals for this project were to build my skills in natural language processing using each supervised learning, unsupervised learning, and data visualization. In supervised learning, I wanted to explore various models to see if I could create a predictor for when the song came out, using a binary classification of pre- and post-2000. In unsupervised learning, I hoped to explore non-negative matrix factorization to see what latent topics I could discover not only in my binary division but on the corpus in its entirety. I wanted to use data visualization to explore things like artist's most frequently used words, as well as the result of my analysis with things like gini importance.

# Data
All data was scraped from [Genius](https://www.genius.com) using [lyricsgenius](https://github.com/johnwmillr/LyricsGenius), a Python client for the Genius.com API. In its entirety, I collected information on 28,816 songs (this data include artist name, song title, album, producer, release data, URL, and lyrics). Most songs contained 400-600 words. When I created my supervised learning models, I used the songs for which I had release dates (18,470 in total). These songs were collected from 100 artists, ranging from early pioneers like Slick Rick, Run-DMC, and Public Enemy, to more recent artists like Drake, 2 Chainz, and Migos.

# Exploratory Data Analysis

# Results

# Other Factors to Consider

# Technologies
* Python (including Seaborn, Matplotlib, Pandas, Numpy, Scikit-learn, SpaCy, and Imbalanced-learn)
* Amazon EC2

# Looking Forward
I'd like to continue exploring the data I have collected in a number of different ways:

* Looking into creating a multi-class predictor for decades as opposed to pre/post-2000.
* Exploring other models that I'm currently not as familiar with, such as XGBoost, support vector machines, and neural networks (I'm especially interested in neural networks as I believe they could give me kind of specificity/accuracy I'd be interested in)
* Placing a gap in time between the two classes. I split on, and included, the year 2000, but I'd be interested in seeing if a gap more like 1975 - 1995 & 2000-2020 would yield better results
* Collecting more data from rappers prior to the 2000s. The bulk of my data came post-2000s, though this was for lack of trying. Current rappers are more prolific than ever. For reference, the people with the most songs in my dataset were Lil Wayne (1131), Gucci Mane (1127), and Chief Keef (792), and those with the fewest were Slick Rick (69), Heavy D (56), and MC Hammer (46) (Cardi B had 54 but she's just getting started).

# Acknowledgements
A big thanks to Dan Rupp, Juliana Duncan, Peter Galea, and Austin Penner, each of whom poured a lot of their time and energy in helping me complete this project. A special thanks, too, to [Genius](https://www.genius.com) which is where I received all data used for this project. The wrapper used to scrape my data, [lyricsgenius](https://github.com/johnwmillr/LyricsGenius), was created by [John Miller](https://github.com/johnwmillr).
