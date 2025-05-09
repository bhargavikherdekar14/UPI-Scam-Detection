from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
import MySQLdb
from sklearn.preprocessing import LabelEncoder
import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Email configuration
EMAIL_SENDER = os.getenv('EMAIL_SENDER')  
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')  

def send_email(to_email, subject, body):
    em = EmailMessage()
    em['From'] = EMAIL_SENDER
    em['To'] = to_email
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(em)

# MySQL configuration
app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'abc@123'
app.config['MYSQL_DB'] = 'upi_scam_detection'

# MySQL connection
mysql = MySQLdb.connect(host=app.config['MYSQL_HOST'],
                        user=app.config['MYSQL_USER'],
                        passwd=app.config['MYSQL_PASSWORD'],
                        db=app.config['MYSQL_DB'])
cursor = mysql.cursor()

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

state_encoder = LabelEncoder()
state_encoder.classes_ = pickle.load(open('state_encoder.pkl', 'rb'))

seller_encoder = LabelEncoder()
seller_encoder.classes_ = pickle.load(open('seller_encoder.pkl', 'rb'))

merchant_encoder = LabelEncoder()
merchant_encoder.classes_ = pickle.load(open('merchant_encoder.pkl', 'rb'))

def safe_transform(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else -1

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'register' in request.form:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            cursor.execute("SELECT * FROM user WHERE email=%s", (email,))
            if cursor.fetchone():
                flash("Email already registered.", "error")
            else:
                cursor.execute("INSERT INTO user (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
                mysql.commit()
                try:
                    send_email(email, "Registration Successful",
                            f"Hi {name}, you have successfully registered for the UPI Scam Detection System.")
                except Exception as e:
                    print("Email sending failed:", e)
                flash("Registration successful. Please login.", "success")

        elif 'login' in request.form:
            email = request.form['email']
            password = request.form['password']
            cursor.execute("SELECT * FROM user WHERE email=%s AND password=%s", (email, password))
            user = cursor.fetchone()
            if user:
                session['user'] = email
                session['username'] = user[1]  
                return redirect(url_for('form'))
            else:
                flash("Invalid email or password", "error")

    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if 'user' not in session:
        return redirect(url_for('index'))

    result = None
    prob = None
    show_result = False
    if request.method == 'POST':
        try:
            upi_number = request.form['upi_number']
            upi_holder_name = request.form['upi_holder_name']
            state = request.form['state']
            pin_code = request.form['pin_code']
            datetime_str = request.form['datetime']
            transaction_amount = float(request.form['transaction_amount'])
            seller_name = request.form['seller_name']
            merchant_category = request.form['merchant_category']

            # Convert datetime
            dt = pd.to_datetime(datetime_str)
            hour = dt.hour
            day_of_week = dt.dayofweek

            # Encode
            state_encoded = safe_transform(state_encoder, state)
            seller_encoded = safe_transform(seller_encoder, seller_name)
            merchant_encoded = safe_transform(merchant_encoder, merchant_category)

            features = pd.DataFrame([{
                'state': state_encoded,
                'pin_code': int(pin_code),
                'transaction_amount': transaction_amount,
                'seller_name': seller_encoded,
                'merchant_category': merchant_encoded,
                'hour': hour,
                'day_of_week': day_of_week
            }])

            scaled_features = scaler.transform(features)

            # Get prediction probabilities
            prob = model.predict_proba(scaled_features)[0][1] 

            # Predict class (scam or valid)
            prediction = model.predict(scaled_features)[0]
            result = "Suspicious Transaction" if prediction == 1 else "Transaction Is Likely"
            show_result = True  

            # Insert transaction into the database
            cursor.execute(""" 
                INSERT INTO transaction (upi_number, state, pin_code, transaction_amount, seller_name, merchant_category, result)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (upi_number, state, pin_code, transaction_amount, seller_name, merchant_category, result))
            mysql.commit()

            # Send result email
            try:
                send_email(session['user'], "Transaction Prediction Result",
                        f"Hello {session['username']},\n\nThe transaction you submitted is: {result}\n\n"
                           f"Prediction Probability: {prob * 100:.2f}%")
            except Exception as e:
                print("Email sending failed:", e)

        except Exception as e:
            print("Error:", e)
            flash("Invalid input. Please try again.", "error")
        
        # Make sure the response is always returned
        return render_template('index.html', result=result, prob=prob, show_form=True, username=session.get('username'), scam_probability=prob, show_result=show_result)
    
    # Ensure there's a return statement for the GET request
    return render_template('index.html', show_form=True, username=session.get('username'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)



