from flask import Flask, render_template, flash, request, jsonify, Markup
import pandas as pd
import joblib

app = Flask(__name__)


# Textbook Interface
# model constants
# set up constants for our coefficients 
INTERCEPT = -121.029547
COEF_HOLIDAY = -23.426176   # if day is holiday or not
COEF_HOUR = 8.631624        # hour (0 to 23)
COEF_SEASON_1 = 3.861149    # 1:spring 
COEF_SEASON_2 = -1.624812   # 2:summer
COEF_SEASON_3 = -41.245562  # 3:fall
COEF_SEASON_4 = 39.009224   # 4:winter
COEF_TEMP = 426.900259      # normalized temp in Celsius -8 to +39

# mean values
MEAN_HOLIDAY = 0.0275   # day is holiday or not
MEAN_HOUR = 11.6        # hour (0 to 23)
MEAN_SEASON_1 = 1       # 1:spring
MEAN_SEASON_2 = 0       # 2:summer
MEAN_SEASON_3 = 0       # 3:fall
MEAN_SEASON_4 = 0       # 4:winter
MEAN_TEMP = 0.4967      # norm temp in Celsius -8 to +39


# My Interface
def data_processing_predict(HR, SEASON, HOLIDAY, TEMP):
    data = [HR, SEASON, HOLIDAY, TEMP]
    data = pd.DataFrame({'hr': [HR], 'season': [SEASON], 'holiday': [HOLIDAY], 'temp': [TEMP]})

    # open file
    file = open("final_Model.pkl", "rb")

    # load trained model
    trained_model = joblib.load(file)

    # predict
    prediction = trained_model.predict(data)

    return round(prediction[0], 0)

@app.route('/')
@app.route('/index', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        # get form data
        HR = request.form.get('hr')
        SEASON = request.form.get('season')
        HOLIDAY = request.form.get('holiday')
        TEMP = request.form.get('temp')

        # call preprocessDataAndPredict and pass inputs
        try:
            prediction = data_processing_predict(HR, SEASON, HOLIDAY, TEMP)
            # pass prediction to template
            return render_template("index.html", prediction=prediction)
        except ValueError:
            return "Please Enter valid values"

        pass
    pass

@app.route("/textbook_userInterface", methods=['POST', 'GET'])
def textbook_version():
    # on load set form with defaults
    return render_template('textbook_userInterface.html',
            mean_holiday = MEAN_HOLIDAY,
            mean_hour = MEAN_HOUR,
            mean_sesaon1 = MEAN_SEASON_1,
            mean_sesaon2 = MEAN_SEASON_2,
            mean_sesaon3 = MEAN_SEASON_3,
            mean_sesaon4 = MEAN_SEASON_4,
            mean_temp = MEAN_TEMP,
            model_intercept = INTERCEPT,
            model_holiday = COEF_HOLIDAY,
            model_hour = COEF_HOUR,
            model_season1 = COEF_SEASON_1,
            model_season2 = COEF_SEASON_2,
            model_season3 = COEF_SEASON_3,
            model_season4 = COEF_SEASON_4,
            model_temp = COEF_TEMP)

if __name__ == '__main__':
    app.run(debug=True)




