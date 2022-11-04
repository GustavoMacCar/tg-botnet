import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
import sys

#Preparing the data

#file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
#dataframe = pd.read_csv(file_url)

dataframe = pd.read_csv('ctu_13_labeled_output.csv')
del dataframe["StartTime"]
del dataframe["SrcAddr"]
del dataframe["Sport"]
del dataframe["DstAddr"]
del dataframe["Dport"]

#print(dataframe.shape)

#print(dataframe.head())

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)


print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("Botnet")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)


train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)


#feature preprocessing with keras layers

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

#build a model

# Categorical features encoded as integers
# sex = keras.Input(shape=(1,), name="sex", dtype="int64")
# cp = keras.Input(shape=(1,), name="cp", dtype="int64")
# fbs = keras.Input(shape=(1,), name="fbs", dtype="int64")
# restecg = keras.Input(shape=(1,), name="restecg", dtype="int64")
# exang = keras.Input(shape=(1,), name="exang", dtype="int64")
# ca = keras.Input(shape=(1,), name="ca", dtype="int64")

#Botnet = keras.Input(shape=(1,), name="Botnet", dtype="int64")

# Categorical feature encoded as string
#thal = keras.Input(shape=(1,), name="thal", dtype="string")
Proto = keras.Input(shape=(1,), name="Proto", dtype="string")
Dir = keras.Input(shape=(1,), name="Dir", dtype="string")
State = keras.Input(shape=(1,), name="State", dtype="string")

# Numerical features
# age = keras.Input(shape=(1,), name="age")
# trestbps = keras.Input(shape=(1,), name="trestbps")
# chol = keras.Input(shape=(1,), name="chol")
# thalach = keras.Input(shape=(1,), name="thalach")
# oldpeak = keras.Input(shape=(1,), name="oldpeak")
# slope = keras.Input(shape=(1,), name="slope")
Dur = keras.Input(shape=(1,), name="Dur")
TotPkts = keras.Input(shape=(1,), name="TotPkts")
TotBytes = keras.Input(shape=(1,), name="TotBytes")
SrcBytes = keras.Input(shape=(1,), name="SrcBytes")
sTos = keras.Input(shape=(1,), name="sTos")
dTos = keras.Input(shape=(1,), name="dTos")


# all_inputs = [
#     sex,
#     cp,
#     fbs,
#     restecg,
#     exang,
#     ca,
#     thal,
#     age,
#     trestbps,
#     chol,
#     thalach,
#     oldpeak,
#     slope,
# ]


all_inputs = [
    Proto,
    Dir,
    State,
    Dur,
    TotPkts,
    TotBytes,
    SrcBytes,
    sTos,
    dTos,
]

# Integer categorical features
# sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
# cp_encoded = encode_categorical_feature(cp, "cp", train_ds, False)
# fbs_encoded = encode_categorical_feature(fbs, "fbs", train_ds, False)
# restecg_encoded = encode_categorical_feature(restecg, "restecg", train_ds, False)
# exang_encoded = encode_categorical_feature(exang, "exang", train_ds, False)
# ca_encoded = encode_categorical_feature(ca, "ca", train_ds, False)

# String categorical features
#thal_encoded = encode_categorical_feature(thal, "thal", train_ds, True)
Proto_encoded = encode_categorical_feature(Proto, "Proto", train_ds, True)
Dir_encoded = encode_categorical_feature(Dir, "Dir", train_ds, True)
State_encoded = encode_categorical_feature(State, "State", train_ds, True)


# Numerical features
# age_encoded = encode_numerical_feature(age, "age", train_ds)
# trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
# chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
# thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
# oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
# slope_encoded = encode_numerical_feature(slope, "slope", train_ds)
Dur_encoded = encode_numerical_feature(Dur, "Dur", train_ds)
TotPkts_encoded = encode_numerical_feature(TotPkts, "TotPkts", train_ds)
TotBytes_encoded = encode_numerical_feature(TotBytes, "TotBytes", train_ds)
SrcBytes_encoded = encode_numerical_feature(SrcBytes, "SrcBytes", train_ds)
sTos_encoded = encode_numerical_feature(sTos, "sTos", train_ds)
dTos_encoded = encode_numerical_feature(dTos, "dTos", train_ds)


all_features = layers.concatenate(
    [
        Proto_encoded,
        Dir_encoded,
        State_encoded,
        Dur_encoded,
        TotPkts_encoded,
        TotBytes_encoded,
        SrcBytes_encoded,
        sTos_encoded,
        dTos_encoded,
    ]
)
x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

# `rankdir='LR'` is to make the graph horizontal.
keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

#train the model
model.fit(train_ds, epochs=50, validation_data=val_ds)

#Inference on new data
# sample = {
#     "age": 60,
#     "sex": 1,
#     "cp": 1,
#     "trestbps": 145,
#     "chol": 233,
#     "fbs": 1,
#     "restecg": 2,
#     "thalach": 150,
#     "exang": 0,
#     "oldpeak": 2.3,
#     "slope": 3,
#     "ca": 0,
#     "thal": "fixed",
# }

sample = {
    "Proto": "tcp",
    "Dir": "->",
    "State": "SR_SA",
    "Dur": 1.026539,
    "TotPkts": 4,
    "TotBytes": 276,
    "SrcBytes": 156,
    "sTos": 0.0,
    "dTos": 0.0,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)

print(
    f"Predictions[0][0]: {(predictions[0][0],)}"
)
