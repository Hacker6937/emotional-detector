import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# load the pre-trained model for emotion detection
model = model_from_json(open("emotion_model.json", "r").read())
model.load_weights('emotion_model.h5')

# define the emotions
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# start the video capture
cap = cv2.VideoCapture(0)

while True:
    # read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the frame using Haar cascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # for each detected face
    for (x,y,w,h) in faces:
        # draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # extract the region of interest (the face)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        # use the pre-trained model to predict the emotion
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]

        # write the predicted emotion above the rectangle
        cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # display the resulting frame
    cv2.imshow('frame',frame)

    # exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()