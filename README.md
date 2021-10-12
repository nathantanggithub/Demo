#####   requirements.txt   #####
You can use the configuration file requirements.txt to install the specified packages with the specified version.

#####   1. Machine Learning   #####
Step 1: Import data from Yahoo by yahoo_data.py

Step 2: Create or Modify features in the .csv file (generated from Step 1) via features_creation.q (kdb+) 
e.g. To run terminal command: q /Users/lapkeitang/Documents/PycharmProjects/Nathan_Demonstration/features_creation.q -q

Step 3: cross validation on forecasting model with either forecasting_lstm_and_others.py or forecasting_lstm_encoder.py

#####   2. Web Scraping   #####
An example is provided on web scraping through web_scraping.py
Remark: It will take around 2 minutes and some errors log due to problematic URLs are expected, the traceback is only for your reference. 