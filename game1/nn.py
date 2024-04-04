import keras
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from game import controlled_run, DO_NOTHING, JUMP

# Variables
total_number_of_games = 50
games_count = 0

# Neural network training data
x_train = []
y_train = []

really_huge_number = 1000

# Training parameters
train_frequency = 10
average_score_rate = 10

model_path = 'model.keras'
# Load or create model
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully from:", model_path)
else:
    model = Sequential([
        LSTM(units=16, input_shape=(1, 1), activation='tanh', return_sequences=False),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
    print("New model created.")

# Visualization setup
fig, _ = plt.subplots(ncols=1, nrows=3, figsize=(6, 6))
fig.tight_layout()

all_scores = []
average_scores = []
all_x, all_y = np.array([]), np.array([])


class Wrapper(object):
    def __init__(self):
        controlled_run(self, 0)

    @staticmethod
    def visualize():
        global all_x, all_y, average_scores, all_scores, x_train, y_train

        plt.subplot(3, 1, 1)
        x = np.linspace(1, len(all_scores), len(all_scores))
        plt.plot(x, all_scores, 'o-', color='r')
        plt.xlabel("Games")
        plt.ylabel("Score")
        plt.title("Score per game")

        plt.subplot(3, 1, 2)
        plt.scatter(x_train[y_train == 0], y_train[y_train == 0], color='r', label='Stay still')
        plt.scatter(x_train[y_train == 1], y_train[y_train == 1], color='b', label='Jump')
        plt.xlabel('Distance from the nearest enemy')
        plt.title('Training data')

        plt.subplot(3, 1, 3)
        x2 = np.linspace(1, len(average_scores), len(average_scores))
        plt.plot(x2, average_scores, 'o-', color='b')
        plt.xlabel("Games")
        plt.ylabel("Score")
        plt.title("Average scores per 10 games")

        plt.pause(0.001)

    def control(self, values):
        global x_train, y_train, games_count, model

        if values['closest_enemy'] == -1:
            return DO_NOTHING

        if values['old_closest_enemy'] != -1 and values['score_increased'] == 1:
            x_train.append(values['old_closest_enemy'] / really_huge_number)
            y_train.append(values['action'])

        prediction = model.predict(np.array([[values['closest_enemy'] / really_huge_number]]))
        predicted_class = np.argmax(prediction)

        if np.random.rand() < 0.5 * (1 - games_count / 50):
            return 1 - predicted_class
        else:
            return predicted_class

    def gameover(self, score):
        global games_count, x_train, y_train, model, all_x, all_y, all_scores, average_scores

        games_count += 1
        all_x = np.append(all_x, x_train)
        all_y = np.append(all_y, y_train)
        all_scores.append(score)
        Wrapper.visualize()


        if games_count != 0 and games_count % average_score_rate == 0:
            average_score = sum(all_scores[-average_score_rate:]) / average_score_rate
            average_scores.append(average_score)

        if games_count != 0 and games_count % train_frequency == 0:
            y_train_cat = keras.utils.to_categorical(y_train, num_classes=2)
            model.fit(np.array(x_train).reshape(-1, 1, 1), y_train_cat, epochs=50, verbose=1, shuffle=True)
            x_train, y_train = [], []

            # Save model
            print("Saving model to:", model_path)
            save_model(model_path)
            print("Model saved successfully!")

        if games_count >= total_number_of_games:
            return

        controlled_run(self, games_count)

if __name__ == '__main__':
    w = Wrapper()