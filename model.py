from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=10, activation="softmax"))

    model.compile(optimizer=Adam(), 
                loss="categorical_crossentropy",
                metrics=["accuracy"])
    
    return model