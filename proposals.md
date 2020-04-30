# 1 - Predicting Which NBA Rookies Will Become All-Stars (my "easy" choice)
High Level Description
* I'm interested in taking this year's NBA rookie class and predicting which ones are mostly likely to become All-Stars at some point in their careers, based on historical statistical profiles of such players. The idea would be to feed the model plenty of historical data on players, both All-Stars and non-All-Stars, and see how well I could train it.

Should this be too complex, my backup plan would be to use it on any season period (as opposed to trying to project data from rookies who often have very inconsistent stats due to lower playing time, generally not being as good as other professionals, etc.). This model could also pivot to instead provide predictions for having an MVP-calibur season instead (that is, if I fed my model the stats from this current (past?) season for Lebron and Giannis, could it give me the probability that either is an MVP based on historical data? Or could it tell me that one has a higher probability of being an MVP?)

Your Approach
* In an ideal world I would like to do feature engineering to see if I can make a logistic regression work. This type of model is of specific interest because I think it's coefficients are very informative, especially in the context of seeing if one stat matters more than any other, and if so by how much. Should this fail, I'd likely look into a random forest or gradient boost.

How will people interact with your work
* It would give insight into basketball players and their profiles, perhaps statistics that serve as the best indicators. I think I could present data cleanly in notebooks. In a perfect world I could create a web app where you select a player and a season and it spit back the strength of that player's statistical profile

What are your data sources
* There are a lot of basketball statistical datasets available, and if necessary I know I can scrape this data off of basketballreference.

# 2 - Name that Pup (my "medium" choice)
High Level Description
* The idea is to make a model that will predict a dog's breed based on a picture of it. I would train a neural network on dog images with the breed of dog being the target.

Your Approach
* I would use neural networks for image prediction.

How will people interact with your work
* Some type of interactive app like a Flask dashboard would be cool. People could upload dog images and the machine spit out a guess. Ideally I'd be able to feed the model a ton of dog data that represents most dogs (at minimum the 10-20 most popular).

What are your data sources
* https://www.kaggle.com/c/dog-breed-identification is where I received this idea. This could also be gathered from Google Images perhaps? I'd need to do more research on training a neural network appropriately on image data.

# 3 - What Does a Grammy-Award Winning Album Sound Like? (My 'difficult' choice)
High Level Description
* The loose idea here is that I would like to take lyrics from albums that have won Grammys and use this to determine what the academy thinks such an album sounds like. Can I detect biases? Maybe I can feed my model albums that were nominated and didn't win?

Should this be too complex, I'd still like to do some type of analysis on language in lyrics. Something like sentiment analysis for the top songs on the billboard (how sad/happy are the most popular songs), or instead seeing if a model could take song lyrics and make a prediction on the genre or even the year/decade it came out.

Your Approach
* This would likely be based around NLP, but I don't currently have a strong enough understanding of that process to further elaborate.

How will people interact with your work
* This could be presented neatly in a notebook. I think it would be difficult to do some type of interactive thing given the nature of song lyrics but it'd be cool to further explore.

What are your data sources
* genius.com is a lyric website that I'm pretty sure has APIs. If not, webscraping.
