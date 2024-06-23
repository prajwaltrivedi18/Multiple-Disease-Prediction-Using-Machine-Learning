from flask import Flask, render_template, request, redirect, url_for
from markupsafe import Markup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the diabetes dataset
df_diabetes = pd.read_csv(r'diabete.csv')

# Preprocess the diabetes dataset
X_diabetes = df_diabetes[['Age', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush',
                          'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness',
                          'Alopecia', 'Obesity']]
y_diabetes = df_diabetes['Outcome']

# Split the diabetes dataset into training and test sets
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes, test_size=0.3, stratify=y_diabetes, random_state=2)


# Train the SVM model for diabetes prediction
svm_model_diabetes = SVC(kernel='linear', random_state=2)
svm_model_diabetes.fit(X_train_diabetes, y_train_diabetes)

y_pred_diabetes = svm_model_diabetes.predict(X_test_diabetes)
test_accuracy_diabetes = accuracy_score(y_test_diabetes, y_pred_diabetes)
print("Test Accuracy for Diabetes Prediction:", test_accuracy_diabetes * 100)

# Load the mental health dataset
df_mental_health = pd.read_csv(r'mental.csv')

# Separate features and target variable for mental health dataset
X_mental_health = df_mental_health.drop('Expert Diagnose', axis=1)
y_mental_health = df_mental_health['Expert Diagnose']

# Encode categorical variables for mental health dataset
label_encoders_mental_health = {}
for column in X_mental_health.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_mental_health[column] = le.fit_transform(X_mental_health[column])
    label_encoders_mental_health[column] = le

# Handle missing values if any for mental health dataset
X_mental_health = X_mental_health.fillna(X_mental_health.mean())

rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
# gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the individual classifiers
rf_classifier.fit(X_mental_health, y_mental_health)
# gb_classifier.fit(X_mental_health, y_mental_health)

# Initialize a voting classifier with the individual classifiers for mental prediction
# voting_classifier_m= VotingClassifier(estimators=[('rf', rf_classifier), ('gb', gb_classifier)], voting='soft')
# voting_classifier_m.fit(X_mental_health, y_mental_health)

# Split the mental health dataset into training and testing sets
X_train_mental_health, X_test_mental_health, y_train_mental_health, y_test_mental_health = train_test_split(X_mental_health, y_mental_health, test_size=0.3, random_state=42)

# Train the model for mental health prediction
model_mental_health = RandomForestClassifier()
model_mental_health.fit(X_train_mental_health, y_train_mental_health)

# Calculate accuracy on the test set for mental health prediction
# y_pred_mental_health = voting_classifier_m.predict(X_test_mental_health)
y_pred_mental_health = model_mental_health.predict(X_test_mental_health)
test_accuracy_mental_health = accuracy_score(y_test_mental_health, y_pred_mental_health)
print("Test Accuracy for Mental Health Prediction:", test_accuracy_mental_health * 100)

# Load the heart disease dataset
data = pd.read_csv(r'heart.csv')
data['age'] = data['age'] // 365
X_heart = data.drop(columns=['cardio'])
y_heart = data['cardio']
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.5, random_state=42)
label_encoders_heart = {}
for column in X_heart.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_heart[column] = le.fit_transform(X_heart[column])
    label_encoders_heart[column] = le

# Handle missing values if any for heart dataset
X_heart = X_heart.fillna(X_heart.mean())
# Initialize individual classifiers for heart disease prediction
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
gb_classifier = GradientBoostingClassifier(n_estimators=500, random_state=42)

# Train the individual classifiers
rf_classifier.fit(X_train_heart, y_train_heart)
gb_classifier.fit(X_train_heart, y_train_heart)

# Initialize a voting classifier with the individual classifiers for heart disease prediction
voting_classifier = VotingClassifier(estimators=[('rf', rf_classifier), ('gb', gb_classifier)], voting='soft')
voting_classifier.fit(X_train_heart, y_train_heart)

# Make predictions on the test set for heart disease prediction
y_pred_heart = voting_classifier.predict(X_test_heart)
test_accuracy_heart = accuracy_score(y_test_heart, y_pred_heart)
print("Test Accuracy for Heart Disease Prediction:", test_accuracy_heart * 100)


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/mental_health')
def mental_health():
    return render_template('mental_health.html')

@app.route('/heart_disease')
def heart():
    return render_template('heart.html')

@app.route('/hospitals')
def hospitals():
    return render_template('hospitals.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = 'Diabetes Prediction'
    if request.form['predict_type'] == 'diabetes':
        input_data = []

        # Extract Age
        input_data.append(int(request.form['age']))

        # Extract other symptoms
        input_data.extend([1 if request.form[symptom] == 'Yes' else 0 for symptom in
                           ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush',
                            'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                            'muscle stiffness', 'Alopecia', 'Obesity']])

        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data], columns=['Age', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
                                                       'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
                                                       'Irritability', 'delayed healing', 'partial paresis',
                                                       'muscle stiffness', 'Alopecia', 'Obesity'])

        

        # Predict using the SVM model
        prediction = svm_model_diabetes.predict(input_df)

        # Determine the prediction result
        if prediction[0] == 1:
           mh_result='Risk of Diabetes(ಮಧುಮೇಹದ ಅಪಾಯ).'
           suggestions = """
                <ul>
                    <li><strong>Test your blood sugar often(ನಿಮ್ಮ ರಕ್ತದಲ್ಲಿನ ಸಕ್ಕರೆಯನ್ನು ಆಗಾಗ್ಗೆ ಪರೀಕ್ಷಿಸಿ).</strong></li>

                    <li><strong>Pay attention to your feet (ನಿಮ್ಮ ಪಾದಗಳಿಗೆ ಗಮನ ಕೊಡಿ).</strong></li>

                    <li><strong>Regular Healthcare Visits(ನಿಯಮಿತ ಆರೋಗ್ಯ ಭೇಟಿಗಳು).</strong></li>
                   <br>
                    <li><strong> <a href="https://www.medicalnewstoday.com/articles/324416"> Click to know more!</a></strong></li>

                    
                </ul> 
                
              """
           suggestions=Markup(suggestions)
           home_r = """
               <ul>
                
                    <li><strong>Neem: Neem might help increase glucose uptake by cells and deposition of glycogen (ಬೇವು ಜೀವಕೋಶಗಳಿಂದ ಗ್ಲೂಕೋಸ್ ಹೀರಿಕೊಳ್ಳುವಿಕೆಯನ್ನು ಹೆಚ್ಚಿಸಲು ಮತ್ತು ಗ್ಲೈಕೋಜೆನ್ ಶೇಖರಣೆಯನ್ನು ಹೆಚ್ಚಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ)</strong></li>
<br>
                    <li><strong> <a href="https://pharmeasy.in/blog/8-effective-herbs-to-lower-blood-sugar/">Click to know more!</a></strong></li>

                    
                </ul> 
               
             """
           home_r = Markup(home_r)
           return render_template('ResYes.html', Result=mh_result, Home_R = home_r, Suggestions = suggestions,Disease_name = title)
        else:
            mh_result = 'No risk of  Diabetes(ಮಧುಮೇಹದ ಅಪಾಯವಿಲ್ಲ).'
            suggT = 'Suggestions'
            suggestions = """
                <ul>
                    <li><strong>Healthy Eating Habits(ಆರೋಗ್ಯಕರ ಆಹಾರ ಪದ್ಧತಿ)</strong></li>

                    <li><strong>Regular Physical Activity(ನಿಯಮಿತ ದೈಹಿಕ ಚಟುವಟಿಕೆ)</strong></li>

                    <li><strong>Weight Management(ತೂಕ ನಿರ್ವಹಣೆ)</strong></li>

                    <li><strong>Stay Hydrated(ಹೈಡ್ರೇಟೆಡ್ ಆಗಿರಿ)</strong></li>

                </ul> 
                  """
            precautionary_m="""
                      <ul>
                     <li><strong>Limit red meat and avoid processed meat; choose nuts, beans, whole grains, poultry, or fish instead(ಕೆಂಪು ಮಾಂಸವನ್ನು ಮಿತಿಗೊಳಿಸಿ ಮತ್ತು ಸಂಸ್ಕರಿಸಿದ ಮಾಂಸವನ್ನು ತಪ್ಪಿಸಿ; ಬದಲಿಗೆ ಬೀಜಗಳು, ಬೀನ್ಸ್, ಧಾನ್ಯಗಳು, ಕೋಳಿ ಅಥವಾ ಮೀನುಗಳನ್ನು ಸೇವಿಸಿ).</strong></li>
<br>
                     <li><strong><a href="https://www.mayoclinic.org/diseases-conditions/diabetes/in-depth/diabetes-management/art-20045803">Click to know more!</a></strong></li>
                </ul>
                    """
            precautionary_m=Markup(precautionary_m)
            suggestions = Markup(suggestions)
            return render_template('ResNo.html', Result=mh_result, SuggestionsT=suggT, Suggestions=suggestions, precautionary = precautionary_m,Disease_name = title)

            
        # Redirect to the appropriate page based on the prediction result
        return redirect(url_for('advice', result=result, advice=advice, redirect_url=redirect_url))

    elif request.form['predict_type'] == 'mental_health':
        title = "Mental Health Prediction"
        input_data = {column: request.form[column] for column in X_mental_health.columns}

        # Map form data to numerical values directly
        for column, le in label_encoders_mental_health.items():
            input_data[column] = le.transform([input_data[column]])[0]

        # Convert input data to a NumPy array
        input_data = np.array(list(input_data.values())).reshape(1, -1)

        
        # Make prediction
        prediction = model_mental_health.predict(input_data)[0]
        if prediction=='Depression':
            mh_result = "Risk of Depression(ಖಿನ್ನತೆಯ ಅಪಾಯ)."
           
            suggestions = """
                <ul>
                    <li><strong>Seek Professional Help: They can provide an accurate diagnosis and develop a personalized treatment plan for you(ಅವರು ನಿಖರವಾದ ರೋಗನಿರ್ಣಯವನ್ನು ಒದಗಿಸಬಹುದು ಮತ್ತು ನಿಮಗಾಗಿ ವೈಯಕ್ತಿಕ ಚಿಕಿತ್ಸಾ ಯೋಜನೆಯನ್ನು ಅಭಿವೃದ್ಧಿಪಡಿಸಬಹುದು).</strong></li>

                    <br>
                    <li><strong><a href=" https://www.mayoclinic.org/diseases-conditions/depression/in-depth/depression/art-20045943  ">Click to know more!</a></strong></li>
              
                      </ul> 
                  """
            home_r="""
                      <ul>
                    
                    <li><strong>Sunlight Exposure: Spend time outdoors during daylight hours, as sunlight exposure can boost mood and regulate your body's internal clock(ಹಗಲು ಹೊತ್ತಿನಲ್ಲಿ ಹೊರಾಂಗಣದಲ್ಲಿ ಸಮಯ ಕಳೆಯಿರಿ, ಏಕೆಂದರೆ ಸೂರ್ಯನ ಬೆಳಕಿಗೆ ಒಡ್ಡಿಕೊಳ್ಳುವುದರಿಂದ ಮನಸ್ಥಿತಿಯನ್ನು ಹೆಚ್ಚಿಸಬಹುದು ಮತ್ತು ನಿಮ್ಮ ದೇಹದ ಆಂತರಿಕ ಗಡಿಯಾರವನ್ನು ನಿಯಂತ್ರಿಸಬಹುದು).</strong></li>
               <br>
                    <li><strong><a href=" https://www.healthline.com/health/beating-depression-naturally#meditation">Click to know more!</a></strong></li>
              
                </ul>
                    """
            home_r=Markup(home_r)
            suggestions=Markup(suggestions)
            return render_template('ResYes.html', Result=mh_result,Suggestions=suggestions,Home_R=home_r,Disease_name = title)
        elif prediction=='Normal':
            mh_result = "Normal(ಯಾವುದೇ ಮಾನಸಿಕ ಅಸ್ವಸ್ಥತೆ ಇಲ್ಲ)"
            suggT = 'Suggestions'
            suggestions = """
                <ul>
                     <li><strong>Manage Stress: Develop healthy coping mechanisms to manage stress effectively. This may include practicing relaxation techniques such as deep breathing, meditation, or yoga(ಒತ್ತಡವನ್ನು ಪರಿಣಾಮಕಾರಿಯಾಗಿ ನಿರ್ವಹಿಸಲು ಆರೋಗ್ಯಕರ ನಿಭಾಯಿಸುವ ಕಾರ್ಯವಿಧಾನಗಳನ್ನು ಅಭಿವೃದ್ಧಿಪಡಿಸಿ. ಇದು ಆಳವಾದ ಉಸಿರಾಟ, ಧ್ಯಾನ ಅಥವಾ ಯೋಗದಂತಹ ವಿಶ್ರಾಂತಿ ತಂತ್ರಗಳನ್ನು ಅಭ್ಯಾಸ ಮಾಡುವುದನ್ನು ಒಳಗೊಂಡಿರಬಹುದು).</strong></li>

                    
                    </ul> 
               """
            precautionary_m = """
                <ul>
                
                    <li><strong>Boundaries and Time Management: Set boundaries in your personal and professional life to prevent overwhelm and burnout(ವಿಪರೀತ ಮತ್ತು ಭಸ್ಮವಾಗುವುದನ್ನು ತಡೆಯಲು ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಮತ್ತು ವೃತ್ತಿಪರ ಜೀವನದಲ್ಲಿ ಗಡಿಗಳನ್ನು ಹೊಂದಿಸಿ). </strong></li>

                    <br>
                     <li><strong><a href=" https://en.wikipedia.org/wiki/Prevention_of_mental_disorders">Click to know more!</a></strong></li>
              
                </ul> 
                """
            precautionary_m=Markup(precautionary_m)
            suggestions=Markup(suggestions)
            return render_template('ResNo.html', Result=mh_result, SuggestionsT=suggT,precautionary = precautionary_m,Suggestions = suggestions,Disease_name = title)
        elif prediction=='Bipolar Type-1':
            mh_result = " Risk of Bipolar Type-1(manic-depressive disorder/ಉನ್ಮಾದ-ಖಿನ್ನತೆಯ ಅಸ್ವಸ್ಥತೆ)."
            suggestions = """
                <ul>
                    
                    <li><strong>Therapy: Regular therapy sessions, such as cognitive-behavioral therapy (CBT) or interpersonal therapy, can help individuals to manage stress(ಕಾಗ್ನಿಟಿವ್-ಬಿಹೇವಿಯರಲ್ ಥೆರಪಿ (CBT) ಅಥವಾ ಇಂಟರ್ ಪರ್ಸನಲ್ ಥೆರಪಿಯಂತಹ ನಿಯಮಿತ ಚಿಕಿತ್ಸಾ ಅವಧಿಗಳು ಒತ್ತಡವನ್ನು ನಿರ್ವಹಿಸಲು ವ್ಯಕ್ತಿಗಳಿಗೆ ಸಹಾಯ ಮಾಡಬಹುದು). </strong></li>
<br>
                   <li><strong><a href="  https://www.everydayhealth.com/bipolar-disorder/the-most-effective-ways-to-manage-bipolar-1-disorder/ ">Click to know more!</a></strong></li>

                </ul> 
                    """
            home_r="""
                      <ul>
                   
                    <li><strong>Light therapy: The therapy involves timed exposure to periods of light and darkness and regimented sleep and wake times(ಚಿಕಿತ್ಸೆಯು ಬೆಳಕು ಮತ್ತು ಕತ್ತಲೆಯ ಅವಧಿಗಳಿಗೆ ಸಮಯಕ್ಕೆ ಒಡ್ಡಿಕೊಳ್ಳುವುದನ್ನು ಒಳಗೊಂಡಿರುತ್ತದೆ ಮತ್ತು ರೆಜಿಮೆಂಟೆಡ್ ನಿದ್ರೆ ಮತ್ತು ಎಚ್ಚರದ ಸಮಯಗಳನ್ನು ಒಳಗೊಂಡಿರುತ್ತದೆ).</strong></li>
<br>
                    <li><strong><a href="  https://www.medicalnewstoday.com/articles/314435 ">Click to know more!</a></strong></li>

                    </ul>
                     """
            home_r=Markup(home_r)
            suggestions=Markup(suggestions)
            return render_template('ResYes.html', Result=mh_result, Suggestions=suggestions,Home_R=home_r,Disease_name = title)
        elif prediction=='Bipolar Type-2':
            mh_result = "Risk of Bipolar Type-2(manic-depressive disorder/ಉನ್ಮಾದ-ಖಿನ್ನತೆಯ ಅಸ್ವಸ್ಥತೆ)."
            suggestions = """
                <ul>
                    
                    <li><strong>Mood Tracking: Keep a mood journal to track your mood fluctuations, sleep patterns, medication adherence, and any significant life events(ನಿಮ್ಮ ಮೂಡ್ ಏರಿಳಿತಗಳು, ನಿದ್ರೆಯ ಮಾದರಿಗಳು, ಔಷಧಿಗಳ ಅನುಸರಣೆ ಮತ್ತು ಯಾವುದೇ ಮಹತ್ವದ ಜೀವನದ ಘಟನೆಗಳನ್ನು ಪತ್ತೆಹಚ್ಚಲು ಮೂಡ್ ಜರ್ನಲ್ ಅನ್ನು ಇರಿಸಿಕೊಳ್ಳಿ). </strong></li>

                  <br>
                    <li><strong><a href="https://www.mayoclinic.org/diseases-conditions/bipolar-disorder/expert-answers/bipolar-treatment/faq-20058042">Click to know more!</a></strong></li>

                     </ul> 
                 """
            home_r="""
                      <ul>

                    <li><strong>Limit Caffeine and Alcohol: Reduce or avoid caffeine and alcohol, as they can disrupt sleep patterns and exacerbate mood symptoms(ಕೆಫೀನ್ ಮತ್ತು ಆಲ್ಕೋಹಾಲ್ ಅನ್ನು ಕಡಿಮೆ ಮಾಡಿ ಅಥವಾ ತಪ್ಪಿಸಿ, ಏಕೆಂದರೆ ಅವು ನಿದ್ರೆಯ ಮಾದರಿಯನ್ನು ಅಡ್ಡಿಪಡಿಸಬಹುದು ಮತ್ತು ಮನಸ್ಥಿತಿಯ ಲಕ್ಷಣಗಳನ್ನು ಉಲ್ಬಣಗೊಳಿಸಬಹುದು).</strong></li>
<br>
                     <li><strong> <a href=" https://valleyoaks.org/health-hub/natural-remedies-for-treating-bipolar-disorder/">Click to know more!</a></strong></li>
                    </ul>
                    """
            home_r=Markup(home_r)
            suggestions=Markup(suggestions)
            return render_template('ResYes.html', Result=mh_result, Suggestions=suggestions,Home_R=home_r,Disease_name = title)


        
    elif request.form['predict_type'] == 'heart_disease':
        title = 'Cardiovascular Disease Prediction'
        age_days = int(request.form['age'])
        age_years = age_days // 365  # Convert age from days to years
        gender = int(request.form['gender'])
        height = float(request.form['height']) * 30.48
        weight = float(request.form['weight'])
        ap_hi = float(request.form['ap_hi'])
        ap_lo = float(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = 1 if 'smoke' in request.form else 0
        alco = 1 if 'alco' in request.form else 0
        active = 1 if 'active' in request.form else 0

        prediction = voting_classifier.predict([[age_years, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc,
                                                  smoke, alco, active]])

        if prediction[0] == 0:
           mh_result = "No risk of Cardiovascular Disease."
           suggT = 'Suggestions'
           suggestions = """
                <ul>
                    
                      <li><strong>Monitor Blood Pressure and Cholesterol: Monitor your blood pressure and cholesterol levels regularly, as recommended by your healthcare provider(ನಿಮ್ಮ ಆರೋಗ್ಯ ರಕ್ಷಣೆ, ರಕ್ತದೊತ್ತಡ ಮತ್ತು ಕೊಲೆಸ್ಟ್ರಾಲ್ ಮಟ್ಟವನ್ನು ನಿಯಮಿತವಾಗಿ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ). </strong></li>

                   <br>
                     <li><strong><a href=" https://www.ucsfhealth.org/education/heart-healthy-tips">Click to know more!</a></strong></li>
                  
                </ul> 
                 """
           precautionary_m = """
                <ul>

                   
                    <li><strong>Know the Signs of Heart Attack and Stroke: Familiarize yourself with the signs and symptoms of a heart attack or stroke, and seek immediate medical attention (ಹೃದಯಾಘಾತ ಅಥವಾ ಪಾರ್ಶ್ವವಾಯುವಿನ ಚಿಹ್ನೆಗಳು ಮತ್ತು ರೋಗಲಕ್ಷಣಗಳೊಂದಿಗೆ ನೀವೇ ಪರಿಚಿತರಾಗಿರಿ ಮತ್ತು ತಕ್ಷಣದ ವೈದ್ಯಕೀಯ ಆರೈಕೆಯನ್ನು ಪಡೆಯಿರಿ). </strong></li>
<br>
                      <li><strong><a href=" https://www.mayoclinic.org/diseases-conditions/heart-disease/in-depth/heart-disease-prevention/art-20046502">Click to know more!</a></strong></li>
                  
                </ul> 
                 """
           precautionary_m=Markup(precautionary_m)
           suggestions=Markup(suggestions)
           return render_template('ResNo.html', Result=mh_result, SuggestionsT=suggT, precautionary = precautionary_m,Suggestions = suggestions,Disease_name = title)
        else:
            mh_result = 'Risk of Cardiovascular Disease'
            suggestions = """
                <ul>
                    <li><strong>Encourage a Heart-Healthy Diet: Adapt a diet low in saturated fats, trans fats, cholesterol, and sodium,  lean proteins, and healthy fats like those found in fish and nuts(ಸ್ಯಾಚುರೇಟೆಡ್ ಕೊಬ್ಬುಗಳು, ಟ್ರಾನ್ಸ್ ಕೊಬ್ಬುಗಳು, ಕೊಲೆಸ್ಟ್ರಾಲ್ ಮತ್ತು ಸೋಡಿಯಂ, ನೇರ ಪ್ರೋಟೀನ್ಗಳು ಮತ್ತು ಮೀನು ಮತ್ತು ಬೀಜಗಳಲ್ಲಿ ಕಂಡುಬರುವ ಆರೋಗ್ಯಕರ ಕೊಬ್ಬುಗಳಲ್ಲಿ ಕಡಿಮೆ ಆಹಾರವನ್ನು ಅಳವಡಿಸಿಕೊಳ್ಳಿ).</strong></li>

                    <br>
                    <li><strong><a href="  https://www.ucsfhealth.org/education/heart-healthy-tips">Click to know more!</a></strong></li>
                  
                      </ul> 
                 """
            home_r="""
                      <ul>
                    <li><strong>Garlic: Garlic is believed to have heart-protective properties, including lowering blood pressure and cholesterol levels.(ಬೆಳ್ಳುಳ್ಳಿಯು ರಕ್ತದೊತ್ತಡ ಮತ್ತು ಕೊಲೆಸ್ಟ್ರಾಲ್ ಮಟ್ಟವನ್ನು ಕಡಿಮೆ ಮಾಡುವುದು ಸೇರಿದಂತೆ ಹೃದಯ-ರಕ್ಷಣಾತ್ಮಕ ಗುಣಗಳನ್ನು ಹೊಂದಿದೆ ಎಂದು ನಂಬಲಾಗಿದೆ).</strong></li>

                   
                    <br>
                    <li><strong><a href="   https://newsnetwork.mayoclinic.org/discussion/home-remedies-lifestyle-changes-can-help-your-heart-health/">Click to know more!</a></strong></li>
                  
                </ul>
          
                    """
            home_r=Markup(home_r)
            suggestions=Markup(suggestions)
            return render_template('ResYes.html', Result=mh_result, Suggestions=suggestions,Home_R=home_r,Disease_name = title)
        
if __name__ == '__main__':
    app.run(port=8080,debug=True)