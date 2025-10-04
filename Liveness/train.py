# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default='data',
    help="Path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="Path to output loss/accuracy plot")
ap.add_argument("-lr", "--learning_rate", type=float, default=0.0004,
    help="Learning rate for model training")
ap.add_argument("-b", "--batch_size", type=int, default=8,
    help="Batch size for model training")
ap.add_argument("-e", "--epochs", type=int, default=100,
    help="Epochs for model training")
args = vars(ap.parse_args())

# Initialize key variables
INIT_LR = args['learning_rate']
BS = args['batch_size']
EPOCHS = args['epochs']

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32") / 255.0  # 建议dtype直接设成float32

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels, num_classes=2)

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.3, random_state=42)

# Data augmentation
aug = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest"
)

# Compile model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR)  # 修改在这里！✅
model = LivenessNet.build(width=32, height=32, depth=3,
    classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# Train the network
print(f"[INFO] training network for {EPOCHS} epochs...")
H = model.fit(
    x=aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS
)

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=le.classes_
))

# Save the model
model_save_path = '../models/liveness.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"[INFO] Model saved in '{model_save_path}'")

# 设置中文字体为 SimHei（黑体），防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use("ggplot")

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# 绘制 Loss 曲线（平滑版）
plt.figure()
plt.plot(np.arange(0, EPOCHS), smooth_curve(H.history["loss"]), label="训练集损失函数值")
plt.plot(np.arange(0, EPOCHS), smooth_curve(H.history["val_loss"]), label="验证集损失函数值")
plt.title("训练集与验证集的Loss变化曲线")
plt.xlabel("迭代轮数")
plt.ylabel("损失值")
plt.legend(loc="upper right")
plt.grid(True)
loss_plot_path = args["plot"].replace(".png", "_loss_smooth.png")
plt.savefig(loss_plot_path)
plt.close()

# 绘制 Accuracy 曲线（平滑版）
plt.figure()
plt.plot(np.arange(0, EPOCHS), smooth_curve(H.history["accuracy"]), label="训练集准确率")
plt.plot(np.arange(0, EPOCHS), smooth_curve(H.history["val_accuracy"]), label="验证集准确率")
plt.title("训练集与验证集的Accuracy变化曲线")
plt.xlabel("迭代轮数")
plt.ylabel("准确率")
plt.legend(loc="lower right")
plt.grid(True)
acc_plot_path = args["plot"].replace(".png", "_acc_smooth.png")
plt.savefig(acc_plot_path)
plt.close()

