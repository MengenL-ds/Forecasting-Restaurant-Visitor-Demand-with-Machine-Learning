# Forecasting-Restaurant-Visitor-Demand-with-Machine-Learning

This project tackles the Recruit Restaurant Visitor Forecasting challenge released in November of 2017, where the goal is to predict the number of future visitors to restaurants across Japan. Since the target is daily visitor counts, this is framed as a time series forecasting problem.

The dataset is lightweight and beginner-friendly, coming from two major Japanese platforms:
	•	Hot Pepper Gourmet (hpg): similar to Yelp (search and reservations)
	•	AirREGI / Restaurant Board (air): similar to Square (reservation management and POS data)

Data spans from January of 2016 until April of 2017 for training, with the test set covering late April to May 2017. The test period includes Japan’s Golden Week, a national holiday week that heavily impacts restaurant traffic.

**NOTE:** Days when restaurants were closed are excluded from the training set, and test days with closures are not considered in the scoring.
