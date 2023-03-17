# <p align="center">Deployment</p>

![getaround_logo](getaround_logo.png)


## Video presentation

[Bloc 5: Getaround - Video presentation](https://share.vidyard.com/watch/qxAMiumMaxq5VApm2AWm4J?)

## Contact

You can write me at **sheichenkojuly@gmail.com**

## Context 

Getaround is a service where drivers rent cars from owners for a specific time period, from an hour to a few days long. 

When renting a car, clients have to complete a checkin flow at the beginning of the rental and a checkout flow at the end of the rental in order to:
- assess the state of the car and notify other parties of pre-existing damages or damages that occurred during the rental,
- compare fuel levels,
- measure how many kilometers were driven.

The checkin and checkout of the rentals can be done with two major flows:
- Mobile rental agreement on native apps: driver and owner meet and both sign the rental agreement on the owner’s smartphone
- Connect: the driver doesn’t meet the owner and opens the car with his smartphone
(The third possibility, paper contract, is negligible).

At the end of the rental, drivers are supposed to bring back the car on time, but it happens from time to time that they are late for the checkout.

Late returns at checkout can generate high friction for the next driver if the car was supposed to be rented again on the same day : Customer service often reports users unsatisfied because they had to wait for the car to come back from the previous rental or users that even had to cancel their rental because the car wasn’t returned on time.

In order to mitigate those issues it was decided to implement a minimum delta between two rentals. A car won’t be displayed in the search results if the requested checkin or checkout times are too close from an already booked rental.

It solves the late checkout issue but also potentially hurts Getaround/owners revenues: we need to find the right trade off.

The product management team still needs to decide:
- threshold: how long should the minimum delay be?
- scope: should the feature be enabled for all cars?, only Connect cars?


## Goals of the project
 - Create a web dashboard that will help the product management team to answer the above questions
 - Create a documented online API to suggest optimum rental prices for car owners using Machine Learning


## Project structure

The project is organised into four folders:
1. **data** containing the original datasets containing information on driver delays and on rental prices
2. **dashboard** containing analysis of the driver delays dataset, as well as files necessary to create a Streamlit web dashboard
3. **mlflow** folder containing analysis of the rental prices datasets, training scripts for several machine learning models and files necessary to create a MLFlow Tracking web server.
4. **api** folder containg files necessary to create an documented online API.

## Deliverables

- Dashboard on Heroku: [https://getaround-streamlit-15032023.herokuapp.com/](https://getaround-streamlit-15032023.herokuapp.com/)

- MLFlow Server on Heroku: [https://getaround.herokuapp.com/](https://getaround.herokuapp.com/)

- Documented online API for price prediction: [https://getaround-api-15032023.herokuapp.com/](https://getaround-api-15032023.herokuapp.com/)

You can test the API by running the following code in python:

````python

import requests

response = requests.post("https://getaround-api-15032023.herokuapp.com/predict", 
                         json={
                                "model_key": "Peugeot",
                                "mileage": 64832,
                                "engine_power": 135,
                                "fuel": "petrol",
                                "paint_color": "red",
                                "car_type": "convertible",
                                "private_parking_available": True,
                                "has_gps": False,
                                "has_air_conditioning": False,
                                "automatic_car": True,
                                "has_getaround_connect": True,
                                "has_speed_regulator": True,
                                "winter_tires": False
                                }
                                )

print(response.json())

````
Or by running the following command in your terminal:

````bash

curl -X 'POST' \
  'https://getaround-api-15032023.herokuapp.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_key": "Citroën",
  "mileage": 150000,
  "engine_power": 135,
  "fuel": "diesel",
  "paint_color": "black",
  "car_type": "estate",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": false,
  "automatic_car": false,
  "has_getaround_connect": false,
  "has_speed_regulator": false,
  "winter_tires": true
}'
````

## References

- [How to Create a Beautify Tornado Chart with Plotly](https://python.plainenglish.io/how-to-create-a-beautify-tornado-chart-in-python-plotly-6c0519e185b4)
