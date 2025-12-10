from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


class CamApp(App):

    def build(self):
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.model = tf.keras.models.load_model('siamesemodel.keras', custom_objects={'L1Dist':L1Dist})

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    def update(self, *args):
        if not self.capture or not self.capture.isOpened():
            print("Camera is not opened")
            return

        ret, frame = self.capture.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            return
        frame = frame[120:120+250, 200:200+250, :]

        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)

        img = tf.io.decode_jpeg(byte_img)
        
        img = tf.image.resize(img, (100,100))
        img = img / 255.0
        
        return img

    def verify(self, *args):
        detection_threshold = 0.9
        min_required_matches = 10

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')

        if not self.capture or not self.capture.isOpened():
            self.verification_label.text = "Camera not opened"
            return [], False

        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.verification_label.text = "Failed to grab frame"
            return [], False

        h, w, _ = frame.shape
        crop_size = 250
        cx, cy = w // 2, h // 2

        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)

        frame = frame[y1:y2, x1:x2, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        verif_dir = os.path.join('application_data', 'verification_images')

        for image_name in os.listdir(verif_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(os.path.join(verif_dir, image_name))

            pred = self.model.predict(
                list(np.expand_dims([input_img, validation_img], axis=1)),
                verbose=0
            )
            results.append(float(pred))

        if len(results) == 0:
            self.verification_label.text = "No verification images"
            return [], False

        results = np.array(results)

        detection = np.sum(results > detection_threshold)
        verification = detection / len(results)
        verified = detection >= min_required_matches

        self.verification_label.text = 'Verified' if verified else 'Unverified'


        Logger.info(f"results: {results.tolist()}")
        Logger.info(f"detection: {detection}")
        Logger.info(f"verification: {verification}")
        Logger.info(f"verified: {verified}")

        return results, verified

def on_stop(self):
    if self.capture:
        self.capture.release()



if __name__ == '__main__':
    CamApp().run()