# File: /model_training.py
# This file contains the model training code.
# It is responsible for training the model and saving the model weights.
# It also saves the model weights after every epoch.
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import keras_tuner as kt
import joblib

def train_xgboost(X_train_res, y_train_res):
    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_grid, scoring='roc_auc', cv=3, verbose=2)
    grid_search.fit(X_train_res, y_train_res)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_xgb_model = grid_search.best_estimator_

    # Cross-validation scores
    cv_scores = cross_val_score(best_xgb_model, X_train_res, y_train_res, cv=3, scoring='roc_auc')
    print(f"Cross-validation scores: {cv_scores}")

    # Save the model weights
    joblib.dump(best_xgb_model, 'best_xgb_model.pkl')

    return best_xgb_model, best_params

def build_dl_model(hp, input_shape):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=128, step=32), activation='relu', input_dim=input_shape))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_dl(X_train_res, y_train_res):
    input_shape = X_train_res.shape[1]
    # Initialize Keras Tuner
    tuner = kt.RandomSearch(lambda hp: build_dl_model(hp, input_shape), objective='val_accuracy', max_trials=5, executions_per_trial=3, directory='my_dir', project_name='credit_fraud')

    # Search for the best hyperparameters
    tuner.search(X_train_res, y_train_res, epochs=10, validation_split=0.2, verbose=1)
    best_hp = tuner.get_best_hyperparameters()[0]
    best_dl_model = tuner.hypermodel.build(best_hp)

    history = best_dl_model.fit(X_train_res, y_train_res, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

    # Save the model using the .keras extension
    best_dl_model.save('best_dl_model.keras')

    return best_dl_model, best_hp, history
