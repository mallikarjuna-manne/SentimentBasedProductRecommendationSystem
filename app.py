import pandas as pd
from model import SentimentBasedRecommenderSystem
from flask import Flask, request, render_template


app = Flask(__name__)

sent_recomm_model = SentimentBasedRecommenderSystem()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the username as input from the webpage
    user_name = request.form['username'].lower()
    # Fetch recommendation results for the given input
    sent_recomm_output = sent_recomm_model.top_5_recommendations(user_name)
    # Format the data
    res_df = pd.DataFrame({"S.No":[1,2,3,4,5],"ProductName":sent_recomm_output})
    final_output = list(zip(*map(res_df.get, res_df)))
    # Return the response
    if not (sent_recomm_output is None):
        return render_template("index.html", output = final_output, query="Top 5 Product Recommendations")
    else:
        return render_template("index.html", message_display="User Name doesn't exist. Please provide a valid user!")


if __name__ == '__main__':
    app.run()
