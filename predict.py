
import os

import tensorflow as tf

import numpy as np

import pandas as pd

import pickle
# Disable AVX/AVX2 warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

labels = ["Benign", "Malign"]

# Make numpy values easier to read.
np.set_printoptions(precision=6, suppress=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('test.data', header=None)

df[1].replace(to_replace=['M', 'B'], value=[1, 0], inplace=True)

id_dataset = df.pop(0)
diagnosis = df.pop(1)

min_max_scaler = pickle.load(open('scaler.save', 'rb'))

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.transform(df)

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)


dataset_test = tf.data.Dataset.from_tensor_slices((df_normalized.values, diagnosis.values))


model = tf.keras.models.load_model('mymodel.h5')
model.summary()

predictions = model.predict(df_normalized)

for prediction, cancer_type in zip(predictions, list(diagnosis)):
    round_prediction = int(round(prediction[0]))
    percentage = prediction[0] if bool(round_prediction) else (1 - prediction[0])
    predict = labels[1] if bool(round_prediction) else labels[0]
    outcome = labels[1] if bool(cancer_type) else labels[0]
    result = '{:.2%} {}'.format(percentage, predict), 'actual: {}'.format(outcome)
    print(result)

