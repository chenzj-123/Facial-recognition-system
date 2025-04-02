import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
import pickle
from PIL import Image, ImageTk
from ArcFace import loadModel
from sklearn.metrics.pairwise import cosine_similarity

# 初始化模型
model = loadModel()

# 使用 OpenCV 的 Haar 级联模型实现人脸检测替代方案
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 读取人脸特征数据库
with open("face_embeddings.pkl", "rb") as f:
    face_database = pickle.load(f)


def recognize_face(face):
    face = cv2.resize(face, (112, 112))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)
    embedding = model.predict(face)[0]

    # 相似度匹配
    max_sim = 0
    best_match = "未知"
    for name, db_emb in face_database.items():
        sim = cosine_similarity([embedding], [db_emb])[0][0]
        if sim > max_sim and sim > 0.5:
            max_sim = sim
            best_match = name
    return best_match


class RealTimeFaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Keras ArcFace 实时人脸识别")

        self.video_label = Label(self.root)
        self.video_label.pack()

        self.start_button = tk.Button(self.root, text="启动", command=self.start_recognition, bg="green", fg="white")
        self.start_button.pack()

        self.stop_button = tk.Button(self.root, text="退出", command=self.stop_recognition, bg="red", fg="white")
        self.stop_button.pack()

        self.cap = cv2.VideoCapture(0)
        self.running = False

    def start_recognition(self):
        self.running = True
        self.update_frame()

    def stop_recognition(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 使用 Haar 级联进行人脸检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_img = rgb[y:y + h, x:x + w]
                    name = recognize_face(face_img)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeFaceRecognitionApp(root)
    root.mainloop()
