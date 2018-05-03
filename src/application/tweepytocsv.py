#!/usr/bin/python
import tweepy
import csv #Import csv
consumer_key= '5rGoPqkpAnBFFn4D63Mv4MzE3'
consumer_secret= '52QJPwYxdJ2lTlPdpgBhBFDBEojwqLo5UhpJRDBjgx8RW3bQdy'

access_token= '897974522997026816-JoPk4ATHDuWLeOHwch22mhoDt3ckjQo'
access_token_secret='okMzoU5tGtkyNfG2X2zg5bD2GUpgFH1rg7SDK1EdLsR4y'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Open/create a file to append data to
csvFile = open('results2.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,
                           q = "give away-filter:retweets",
                           since = "2018-04-26",
                           until = "2018-04-27",
                           lang = "en").items(2000):

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.created_at, tweet.text.encode('ascii', 'ignore')])
    print(tweet.created_at, tweet.text)
csvFile.close()
