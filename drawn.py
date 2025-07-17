import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from keras.models import load_model

model = load_model("best_model.h5")

root = tk.Tk()
root.title("Simple Draw & Guess")

canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

image = Image.new("L", (280, 280), "white")
draw = ImageDraw.Draw(image)

def paint(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="black")

def predict():
    img = image.resize((28, 28))
    img = ImageOps.invert(img)
    arr = np.array(img)/255.0
    arr = arr.reshape(1, 28, 28, 1)
    pred = model.predict(arr)
    digit = np.argmax(pred)
    conf = np.max(pred)
    label.config(text=f"Prediction: {digit} (Trust: %{conf*100:.2f})")

def clear():
    canvas.delete("all")
    draw.rectangle([0,0,280,280], fill="white")
    label.config(text="")

canvas.bind("<B1-Motion>", paint)

btn_predict = tk.Button(root, text="Predict", command=predict)
btn_predict.pack()

btn_clear = tk.Button(root, text="Clean", command=clear)
btn_clear.pack()

label = tk.Label(root, text="")
label.pack()

root.mainloop()
