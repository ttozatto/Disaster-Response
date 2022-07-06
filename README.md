# Disaster Response Pipeline Project

### Introduction:
This app classifies messages between a wide range of classes. The classes represent different problems that people may face in a disaster, like floods, fire, cold, lack of water and food etc. This tool can help to distinguish the relevant messages during an emergency that invovlves lots of people, this way each responsible institution can direct the right effort to each problem, saving time and resources.


### Disclaimer:
The training data in this repository is imbalanced, it has many samples in some classes and almost zero in others. This app should not be used in real cases with the current ML model, but could be a good starting point to develop a more robust one. This could be achieved with a larger and more uniform training dataset.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run your web app: `python app.py`.

4. Open the homepage at the local address https://127.0.0.1:5000.


![image](https://user-images.githubusercontent.com/42552721/177623830-a09d3dc2-d381-408f-8bc1-73482c8aa131.png)


### Working example:
I deployed an working example of this web app on: https://disaster-response.azurewebsites.net/
(The app uses a free tier hosting, so it can be pretty slow and will probably be shut down at some point.

### Aknowledgements:
I would like to pay my special regards to:

 - Udacity, that proposed this work in the Data Science Nanodegree.
 - Microsoft Azure, that provides a free tier service for web app hosting.
 - NLTK and Scikit-Learn community, that provide amazing python libraries.
