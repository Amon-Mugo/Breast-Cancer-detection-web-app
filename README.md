
Machine learning model on prediction of breast cancer.Also deployed into  UI  using  Flask  where you can test your results and get real time feedback
It is values are maily dependent on medical results in order to determine if the cancer is malignant or not.
Repository: amon-mugo/breast-cancer-detection-web-app
Files analyzed: 3

Estimated tokens: 4.0k

Directory structure:
└── amon-mugo-breast-cancer-detection-web-app/
    ├── README.md
    ├── Breast Cancer detection web app.ipynb
    └── index.html


================================================
FILE: README.md
================================================

Machine learning model on prediction of breast cancer.Also deployed into  UI  using  Flask  where you can test your results and get real time feedback
It is values are maily dependent on medical results in order to determine if the cancer is malignant or not.



================================================
FILE: Breast Cancer detection web app.ipynb
================================================
# Jupyter notebook converted to Python script.

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score

df=pd.read_csv(r"C:\Users\AMON\Desktop\archive (1).zip")
df.head()
# Output:
#            id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \

#   0    842302         M        17.99         10.38          122.80     1001.0   

#   1    842517         M        20.57         17.77          132.90     1326.0   

#   2  84300903         M        19.69         21.25          130.00     1203.0   

#   3  84348301         M        11.42         20.38           77.58      386.1   

#   4  84358402         M        20.29         14.34          135.10     1297.0   

#   

#      smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \

#   0          0.11840           0.27760          0.3001              0.14710   

#   1          0.08474           0.07864          0.0869              0.07017   

#   2          0.10960           0.15990          0.1974              0.12790   

#   3          0.14250           0.28390          0.2414              0.10520   

#   4          0.10030           0.13280          0.1980              0.10430   

#   

#      ...  texture_worst  perimeter_worst  area_worst  smoothness_worst  \

#   0  ...          17.33           184.60      2019.0            0.1622   

#   1  ...          23.41           158.80      1956.0            0.1238   

#   2  ...          25.53           152.50      1709.0            0.1444   

#   3  ...          26.50            98.87       567.7            0.2098   

#   4  ...          16.67           152.20      1575.0            0.1374   

#   

#      compactness_worst  concavity_worst  concave points_worst  symmetry_worst  \

#   0             0.6656           0.7119                0.2654          0.4601   

#   1             0.1866           0.2416                0.1860          0.2750   

#   2             0.4245           0.4504                0.2430          0.3613   

#   3             0.8663           0.6869                0.2575          0.6638   

#   4             0.2050           0.4000                0.1625          0.2364   

#   

#      fractal_dimension_worst  Unnamed: 32  

#   0                  0.11890          NaN  

#   1                  0.08902          NaN  

#   2                  0.08758          NaN  

#   3                  0.17300          NaN  

#   4                  0.07678          NaN  

#   

#   [5 rows x 33 columns]

df.isnull().sum()
# Output:
#   id                           0

#   diagnosis                    0

#   radius_mean                  0

#   texture_mean                 0

#   perimeter_mean               0

#   area_mean                    0

#   smoothness_mean              0

#   compactness_mean             0

#   concavity_mean               0

#   concave points_mean          0

#   symmetry_mean                0

#   fractal_dimension_mean       0

#   radius_se                    0

#   texture_se                   0

#   perimeter_se                 0

#   area_se                      0

#   smoothness_se                0

#   compactness_se               0

#   concavity_se                 0

#   concave points_se            0

#   symmetry_se                  0

#   fractal_dimension_se         0

#   radius_worst                 0

#   texture_worst                0

#   perimeter_worst              0

#   area_worst                   0

#   smoothness_worst             0

#   compactness_worst            0

#   concavity_worst              0

#   concave points_worst         0

#   symmetry_worst               0

#   fractal_dimension_worst      0

#   Unnamed: 32                569

#   dtype: int64

df.drop(columns=['Unnamed: 32'], inplace=True)

df=df.drop(columns=['id'])

selected_features = [
    

    'texture_worst', 'area_se', 'area_worst', 'texture_mean', 'compactness_se',
    'symmetry_worst', 'concavity_worst', 'concave points_worst', 'smoothness_worst', 
    'smoothness_mean','symmetry_se','radius_worst','concavity_mean','area_mean','concave points_se'
 

]

# m=malignant concerous
# b=benign non cancerous
df['status']=df['diagnosis'].map({'M':1,'B':0})
X=df[selected_features]
y=df['status']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=XGBClassifier(use_label_encoder=False,eval_metric='logloss')
model.fit(X_train_scaled,y_train)
# Output:
#   C:\Users\AMON\AppData\Roaming\Python\Python312\site-packages\xgboost\training.py:183: UserWarning: [18:15:17] WARNING: C:\actions-runner\_work\xgboost\xgboost\src\learner.cc:738: 

#   Parameters: { "use_label_encoder" } are not used.

#   

#     bst.update(dtrain, iteration=i, fobj=obj)

#   XGBClassifier(base_score=None, booster=None, callbacks=None,

#                 colsample_bylevel=None, colsample_bynode=None,

#                 colsample_bytree=None, device=None, early_stopping_rounds=None,

#                 enable_categorical=False, eval_metric='logloss',

#                 feature_types=None, feature_weights=None, gamma=None,

#                 grow_policy=None, importance_type=None,

#                 interaction_constraints=None, learning_rate=None, max_bin=None,

#                 max_cat_threshold=None, max_cat_to_onehot=None,

#                 max_delta_step=None, max_depth=None, max_leaves=None,

#                 min_child_weight=None, missing=nan, monotone_constraints=None,

#                 multi_strategy=None, n_estimators=None, n_jobs=None,

#                 num_parallel_tree=None, ...)

from xgboost import plot_importance
plot_importance(model)
# Output:
#   <Axes: title={'center': 'Feature importance'}, xlabel='Importance score', ylabel='Features'>
#   <Figure size 640x480 with 1 Axes>

#evaluation
y_pred=model.predict(X_test_scaled)

print("Acuuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))
print("Recall",recall_score(y_test,y_pred))
# Output:
#   Acuuracy 0.956140350877193

#   Precision 0.9523809523809523

#   Recall 0.9302325581395349


import joblib
joblib.dump(model,'disease_detection.pkl') # save the model to a file
joblib.dump(scaler,'scaler.pkl')
# Output:
#   ['scaler.pkl']

from flask import Flask,request,jsonify,render_template
import numpy as np
import joblib 
model=joblib.load('disease_detection.pkl')
scaler=joblib.load('scaler.pkl')


feature_names = [
    

    'texture_worst', 'area_se', 'area_worst', 'texture_mean', 'compactness_se',
    'symmetry_worst', 'concavity_worst', 'concave points_worst', 'smoothness_worst', 
    'smoothness_mean','symmetry_se','radius_worst','concavity_mean','area_mean','concave points_se'
 

]


app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction',methods=['POST'])
def prediction():
    data=request.get_json()
    try:
        input_data=[float(data[feature])
 for feature in feature_names]
    except KeyError as e:
        return jsonify ({'error':f'Missing feature:{e.args[0]}'}),400
    except ValueError as e:
        return jsonify({'error':f'Invalid value for feature:{str(e)}'}),400
        
    features=np.array(input_data).reshape(1,-1)
    feature_scaled=scaler.transform(features)
    prediction=model.predict(feature_scaled)[0]
    result="Positive" if prediction==1 else "Negative"
    return jsonify({'prediction': int(prediction), 'result': result})
    



if __name__ == '__main__':
    app.run()
# Output:
#    * Serving Flask app '__main__'

#    * Debug mode: off

#   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.

#    * Running on http://127.0.0.1:5000

#   Press CTRL+C to quit

#   127.0.0.1 - - [24/Jul/2025 18:16:17] "GET / HTTP/1.1" 200 -

#   127.0.0.1 - - [24/Jul/2025 18:16:21] "GET /favicon.ico HTTP/1.1" 404 -

#   C:\ProgramData\anaconda3\Lib\site-packages\sklearn\base.py:493: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names

#     warnings.warn(

#   127.0.0.1 - - [24/Jul/2025 18:29:44] "POST /prediction HTTP/1.1" 200 -

#   127.0.0.1 - - [24/Jul/2025 18:30:05] "GET / HTTP/1.1" 200 -

#   C:\ProgramData\anaconda3\Lib\site-packages\sklearn\base.py:493: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names

#     warnings.warn(

#   127.0.0.1 - - [24/Jul/2025 18:35:59] "POST /prediction HTTP/1.1" 200 -


python test_app.py


# Output:
#   Error: SyntaxError: invalid syntax (1957662778.py, line 1)

print(df.columns)
# Output:
#   Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',

#          'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

#          'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

#          'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

#          'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

#          'fractal_dimension_se', 'radius_worst', 'texture_worst',

#          'perimeter_worst', 'area_worst', 'smoothness_worst',

#          'compactness_worst', 'concavity_worst', 'concave points_worst',

#          'symmetry_worst', 'fractal_dimension_worst', 'status'],

#         dtype='object')




================================================
FILE: index.html
================================================

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Breast Cancer Prediction</title>
  <style>
    body {
      margin: 0;
      font-family: 'Segoe UI', sans-serif;
      color: #fff;
      background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
      background-size: 400% 400%;
      animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }


    .container {
      max-width: 800px;
      margin: auto;
      padding: 30px;
      background-color: rgba(0, 0, 0, 0.6);
      border-radius: 20px;
      margin-top: 50px;
    }

    h1 {
      text-align: center;
      font-size: 2em;
      margin-bottom: 30px;
    }

    form {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
    }


    input[type="number"] {
      width: 100%;
      padding: 10px;
      border-radius: 8px;
      border: none;
      font-size: 1em;
    }

    button {
      grid-column: span 2;
      padding: 15px;
      border: none;
      border-radius: 10px;
      background: #23d5ab;
      color: #000;
      font-weight: bold;
      font-size: 1.1em;
      cursor: pointer;
      transition: 0.3s ease;
    }


    button:hover {
      background: #23a6d5;
      color: white;
    }

    .result {
      text-align: center;
      margin-top: 25px;
      font-size: 1.4em;
    }

    /* Loader styles */
    .loader {
      display: none;
      margin: 20px auto;
      border: 8px solid #f3f3f3;
      border-top: 8px solid #23d5ab;
      border-radius: 50%;
      width: 50px;
      height: 50px;

      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🔬 Breast Cancer Prediction</h1>
    <form id="predict-form">
      <!-- Input fields created dynamically -->
    </form>

    <!-- Loader -->
    <div class="loader" id="loader"></div>

    <!-- Result -->
    <div class="result" id="prediction-result"></div>
  </div>

  <script>
    const featureNames = [
      'texture_worst', 'area_se', 'area_worst', 'texture_mean', 'compactness_se',
    'symmetry_worst', 'concavity_worst', 'concave points_worst', 'smoothness_worst', 
    'smoothness_mean','symmetry_se','radius_worst','concavity_mean','area_mean','concave points_se'
 
    ];

    const form = document.getElementById('predict-form');
    const resultDiv = 

document.getElementById('prediction-result');
    const loader = document.getElementById('loader');

    // Create input fields dynamically
    featureNames.forEach(name => {
      const input = document.createElement('input');
      input.type = 'number';
      input.name = name;
      input.placeholder = name.replace(/_/g, ' ');
      input.required = true;
      input.step='any';
      input.inputMode='decimal';
      form.appendChild(input);
    });

    // Submit button
    const button = document.createElement('button');
    button.type = 'submit';

    button.textContent = 'Predict';
    form.appendChild(button);

    // Submit event
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      resultDiv.textContent = '';
      loader.style.display = 'block';

      const formData = new FormData(form);
      const data = {};
      for (const [key, value] of formData.entries()) {
        data[key] = parseFloat(value);
      }

      try {
        const response = await fetch('/prediction', {
          method: 'POST',

          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        const result = await response.json();
        resultDiv.textContent = `🧪 Prediction: ${result.result} (code: ${result.prediction})`;
      } catch (error) {
        resultDiv.textContent = '⚠️ Error occurred during prediction.';
      } finally {
        loader.style.display = 'none';
      }
    });
  </script>
</body>
</html>

