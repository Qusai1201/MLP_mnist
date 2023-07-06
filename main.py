from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from sklearn.model_selection import RandomizedSearchCV
import random

def Preprocess(data):
    resized_images = []
    for image in data:
        image = image / 255
        resized_image = cv2.resize(image.reshape(28, 28), (14, 14))
        resized_images.append(resized_image.flatten())
    return np.array(resized_images)

(X_train, y_train), (_, _) = mnist.load_data()

X_train = Preprocess(X_train)

param_dist = {
    'hidden_layer_sizes': (random.randint(100, 500), random.randint(1 , 4)),
    'activation': ['relu', 'tanh']
}

clf = MLPClassifier()

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best Score: ", random_search.best_score_)
print("Best Params: ", random_search.best_params_)

best_clf = random_search.best_estimator_
pickle.dump(best_clf, open("MLPClassifier_best.pkl", "wb"))