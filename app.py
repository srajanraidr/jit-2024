from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv("test1.csv")
X = data[["N", "P", "K", "PH"]]
y = data["Crops"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=101, train_size=0.7
)
model = SVC()
model.fit(X_train, y_train)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        n = request.form.get("n")
        p = request.form.get("k")
        k = request.form.get("p")
        pH = request.form.get("pH")
        global model
        res = model.predict(np.array([[n, p, k, pH]]))
        print(res)
        return render_template("index.html", res=res[0])

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
