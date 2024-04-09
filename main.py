from flask import Flask , render_template, request, redirect, session, url_for
import os
#from Traffic_Situation_Prediction import Stacking_predict


app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "hello"
    

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
            time = int(request.form['time'])
            car = int(request.form['carcount'])
            bike= int(request.form['bikecount'])
            bus= int(request.form['buscount'])
            truck= int(request.form['truckcount'])
            total= int(request.form['total'])

            #y_pred=Stacking_predict.predict(time,car,bike,bus,truck,total)
            d={"0":"Low","1":"Normal","2":"High","3":"Heavy"}
            #return render_template("index.html", data=d[str(y_pred)])
            return render_template("index.html", data=d[str(3)])
    return render_template("index.html",data="Data will be displayed here")




if __name__ == "__main__":
    app.run(debug=True)
