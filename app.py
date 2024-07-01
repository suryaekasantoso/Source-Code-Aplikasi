from flask import Flask, request, render_template
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Membaca data dari database menggunakan Pandas
data = pd.read_csv('fertility.csv')

# Memisahkan atribut dan label
X = data[['season', 'age', 'child_diseases', 'accident', 'surgical_intervention', 'high_fevers', 'alcohol', 'smoking', 'hrs_sitting']]
y = data['diagnosis']

# Membagi dataset menjadi data latih & data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1, join='inner')
df_test = pd.concat([X_test, y_test], axis=1, join='inner')

# Membuat data sampel bootstrap
B1 = df_train.sample(frac=1)
B2 = df_train.sample(frac=1)
B3 = df_train.sample(frac=1)

# Modelling menggunakan Naive Bayes
B1_x = B1.drop(['diagnosis'], axis=1)
B1_y = B1['diagnosis']

M1 = GaussianNB()
M1.fit(B1_x, B1_y)

M1_predict = M1.predict(X_train)
M1_result = pd.DataFrame(M1_predict, columns=['P1'])

B2_x = B2.drop(['diagnosis'], axis=1)
B2_y = B2['diagnosis']

M2 = GaussianNB()
M2.fit(B2_x, B2_y)

M2_predict = M2.predict(X_train)
M2_result = pd.DataFrame(M2_predict, columns=['P2'])

B3_x = B3.drop(['diagnosis'], axis=1)
B3_y = B3['diagnosis']

M3 = GaussianNB()
M3.fit(B3_x, B3_y)

M3_predict = M3.predict(X_train)
M3_result = pd.DataFrame(M3_predict, columns=['P3'])

# Menggabungkan dan mengubah data ke Numeric
x_combined = pd.concat([M1_result, M2_result, M3_result], axis=1)
X_combined = pd.get_dummies(x_combined, prefix=["P1", "P2", "P3"], columns=["P1", "P2", "P3"], dtype='int')

# Membuat Agregasi
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_combined, y_train)

# Membuat fungsi untuk prediksi menggunakan BaggingClassifier
def baggingClassifier(data):
    B1 = df_train.sample(frac=1)
    B2 = df_train.sample(frac=1)
    B3 = df_train.sample(frac=1)

    B1_x = B1.drop(['diagnosis'], axis=1)
    B1_y = B1['diagnosis']

    M1.fit(B1_x, B1_y)
    M1_predict = M1.predict(data)
    M1_result = pd.DataFrame(M1_predict, columns=['P1'])

    B2_x = B2.drop(['diagnosis'], axis=1)
    B2_y = B2['diagnosis']

    M2.fit(B2_x, B2_y)
    M2_predict = M2.predict(data)
    M2_result = pd.DataFrame(M2_predict, columns=['P2'])

    B3_x = B3.drop(['diagnosis'], axis=1)
    B3_y = B3['diagnosis']

    M3.fit(B3_x, B3_y)
    M3_predict = M3.predict(data)
    M3_result = pd.DataFrame(M3_predict, columns=['P3'])

    x_combined = pd.concat([M1_result, M2_result, M3_result], axis=1)
    x_predict = pd.get_dummies(x_combined, prefix=["P1", "P2", "P3"], columns=["P1", "P2", "P3"], dtype='int')

    # Memastikan kolom-kolom sesuai dengan X_combined
    for col in X_combined.columns:
        if col not in x_predict.columns:
            x_predict[col] = 0
    x_predict = x_predict[X_combined.columns]

    clf_knn.fit(X_combined, y_train)

    # Mengubah output menjadi "O (Altered)" dan "N (Normal)"
    prediction = clf_knn.predict(x_predict)
    prediction = ['O (Altered)' if pred == 'O' else 'N (Normal)' for pred in prediction]

    return prediction, clf_knn.score(X_combined, y_train)

@app.route('/')
def index():
    # Mengirimkan nilai-nilai default jika tidak ada input sebelumnya
    default_values = {
        'season': '-1',
        'age': '',
        'child_diseases': '0',
        'accident': '0',
        'surgical_intervention': '0',
        'high_fevers': '-1',
        'alcohol': '',
        'smoking': '-1',
        'hrs_sitting': ''
    }
    return render_template('index.html', **default_values)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        data = {
            'season': data['season'],
            'age': data['age'],
            'child_diseases': data['child_diseases'],
            'accident': data['accident'],
            'surgical_intervention': data['surgical_intervention'],
            'high_fevers': data['high_fevers'],
            'alcohol': data['alcohol'],
            'smoking': data['smoking'],
            'hrs_sitting': data['hrs_sitting']
        }
        prediction, score = baggingClassifier(pd.DataFrame([list(data.values())], columns=list(data.keys())))
        return render_template('index.html', prediction=prediction[0], accuracy=score, **data)
    except Exception as e:
        return render_template('index.html', error=str(e), **data)

@app.route('/data_info')
def data_info():
    return render_template('data_info.html')

if __name__ == '__main__':
    app.run(debug=True)
