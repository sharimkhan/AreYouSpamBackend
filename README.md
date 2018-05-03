# AreYouSpamBackend
The backend work behind the chrome extension
Files Explained:
extrafiles/results1.txt is the source for the tweets that the ML models read in. 
src/application/analyzer.py is the script that trains different ML models and prints results about them
src/tweepytocsv.py is the script that scrapes twitter for tweets
src/Deployed is the folder that is hosted on Heroku
it contains the Flask app app.py that handles the requests sent to and from the chrome extension
