
from django.shortcuts import render
from .models import *
from .serializers import *
from rest_framework.viewsets import ModelViewSet
from django.core.mail import send_mail, EmailMultiAlternatives, EmailMessage
from django.conf import settings
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
import numpy as np
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as tf_image
from keras.preprocessing import image
import tensorflow as tf
import librosa
from PIL import Image
import os
import string
import random


# Create your views here.

vgg_model = VGG16(weights='imagenet')


drawing_model = tf.keras.models.load_model('drawing.h5')
MRI_model = tf.keras.models.load_model('MRI.h5')
voiceBiometrics_model = tf.keras.models.load_model('Voice.h5')
tappyKeyboard_model = joblib.load('tappy.h5')

print("Model Initialized!")

last_conv_layer_name = 'block5_conv3'


# Define classes for prediction


class ContactEmailViewSet(ModelViewSet):
    queryset = ContactEmails.objects.all()
    serializer_class = ContactEmailsSerializer

    def perform_create(self, serializer):
        result = serializer.save()

        firstName = serializer.data['first_name']
        lastName = serializer.data['last_name']
        email = serializer.data['email']
        subject = serializer.data['subject']
        message = f"Full Name: {firstName} {lastName}\nEmail Address: {email}\n\n{serializer.data['message']}"

        send_mail(subject=subject, message=message, from_email=settings.EMAIL_HOST_USER, recipient_list=[settings.EMAIL_HOST_USER])



class DrawingViewSet(ModelViewSet):
    queryset = Drawing.objects.all()
    serializer_class = DrawingSerializer

    def perform_create(self, serializer):
        result = serializer.save()

        image_url = serializer.instance.image.path

        detection_result = self.drawing_detection(image_url)

        if detection_result >= 0.5:
            result.result = "Parkinson's disease has been detected"
            result.parkinson = True
        else:
            result.result = "The patient does not have Parkinson's disease"

        result.save()

    def drawing_detection(self, image_input):
        img = image.load_img(image_input, target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        prediction = drawing_model.predict(img)

        # Extract the probability score
        return prediction[0][0]




class MRIViewSet(ModelViewSet):
    queryset = MRI.objects.all()
    serializer_class = MRISerializer

    def perform_create(self, serializer):
        result = serializer.save()

        random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

        image_url = serializer.instance.image.path
        gradcam_image, pred_class_name = self.generate_grad_cam(image_url)

        cv2.imwrite(os.path.join(os.getcwd(), f"uploads/heatmap mri/heat_map_{random_id}_{result.image.name.split('/')[-1]}"), cv2.cvtColor(gradcam_image, cv2.COLOR_RGB2BGR))
        heatmap_image_url = self.request.build_absolute_uri(f"/media/heatmap mri/heat_map_{random_id}_{result.image.name.split('/')[-1]}")


        result.heatmap_image = heatmap_image_url
        
        detection_result = self.MRI_detection(image_url)


        if detection_result == 0:
            result.result = "Control, The patient does not have Parkinson's or Alzheimer's disease"
        elif detection_result == 1:
            result.result = "Alzheimer's disease has been detected"
            result.parkinson = True
        elif detection_result == 2:
            result.result = "Parkinson's disease has been detected"
            result.parkinson = True

        result.save()

    def MRI_detection(self, image_input):
        # Preprocess the image for EfficientNetB0
        img = Image.open(image_input)
        img = img.resize((150, 150))
        img = tf_image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Make the prediction using EfficientNetB0
        prediction = MRI_model.predict(img)
        class_idx = np.argmax(prediction, axis=1)[0]
        return class_idx

    def generate_grad_cam(self, image_input):
        # Load and preprocess the image
        img = Image.open(image_input)
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Adjust the size to match VGG16
        img_array = tf_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Preprocess the image for VGG16

        # Get the top predicted class
        preds = vgg_model.predict(img_array)
        predicted_class = np.argmax(preds[0])
        pred_class_name = decode_predictions(preds)[0][0][1]

        # Get the output tensor of the last convolutional layer
        last_conv_layer = vgg_model.get_layer(last_conv_layer_name)

        # Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model([vgg_model.inputs], [last_conv_layer.output, vgg_model.output])

        # Compute the gradient of the top predicted class with respect to the output feature map of the last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, predicted_class]

        grads = tape.gradient(loss, conv_outputs)[0]

        # Compute the CAM
        cam = np.mean(conv_outputs[0], axis=-1)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam / cam.max()

        # Generate heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Superimpose the heatmap on the original image
        superimposed_img = cv2.addWeighted(
            cv2.cvtColor(img_array[0], cv2.COLOR_BGR2RGB).astype('float32'), 0.6,
            heatmap.astype('float32'), 0.4, 0
        )

        return superimposed_img, pred_class_name

        


class VoiceBiometricsViewSet(ModelViewSet):
    queryset = VoiceBiometrics.objects.all()
    serializer_class = VoiceBiometricsSerializer

    def perform_create(self, serializer):
        result = serializer.save()
        # print(serializer.instance.audio.path)

        audio_url = serializer.instance.audio.path
        detection_result = self.voice_detection(audio_url)

        if detection_result == 1:
            result.result = "Your voice has no parkinson's disease"
        else:
            result.result = "Parkinson's disease has been detected"
            result.parkinson = True
        result.save()

    def voice_detection(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512, *args, **kwargs):
        signal, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        features = np.mean(mfccs.T, axis=0)
        prediction = voiceBiometrics_model.predict(np.expand_dims(features, axis=0))
        predicted_class = int(prediction[0][0] > 0.5)
        return predicted_class

class TappyKeyboardViewSet(ModelViewSet):
    queryset = TappyKeyboard.objects.all()
    serializer_class = TappyKeyboardSerializer

    def perform_create(self, serializer):
        result = serializer.save()
        # print(serializer.instance.audio.path)
        file_url = serializer.instance.file.path

        detection_result = self.tappy_detection(file_url)[0]
        print(detection_result)


        if detection_result == 0:
            result.result = "According to file uploaded you have no parkinson's disease"
        else:
            result.result = "Parkinson's disease has been detected"
            result.parkinson = True
        result.save()

    def tappy_detection(self, file_input):
        predictions = []

        df = pd.read_excel(file_input)
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df))
        features_imputed = df_imputed.values.tolist()

        for feature in features_imputed:
            input_data = np.array(feature).astype(np.float64)
            prediction = tappyKeyboard_model.predict(input_data.reshape(1, -1))
            predictions.append(prediction)
            
        return predictions




