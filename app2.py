import streamlit as st
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import numpy as np
import subprocess
from bark import generate_audio
import replicate



mp_holistic = mp.solutions.holistic


# Add the generate_audio function here

def generate_audio(prompt):
    output = replicate.run(
        "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
        input={"prompt": prompt}
    )

    # Check if 'stdout' key is present in the output
    if 'stdout' in output:
        return output['stdout']
    else:
        # If 'stdout' key is not present, return None or an empty string
        return None

    
# Function to perform Mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Function to recognize emotion
def recognize_emotion(frame, face_classifier, emotion_model, emotion_labels):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            prediction = emotion_model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            labels.append(label)
        else:
            labels.append('No Faces')

    return labels

# Function to integrate emotion and sign language
def integrate_emotion_and_sign_language():
    cap = cv2.VideoCapture(0) 
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    with holistic:
        # Load the pre-trained models
        sign_model = load_model('action.h5')
        emotion_model = load_model('model.h5')
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        sequence = []  # Initialize an empty sequence
        recognized_sentence = ""  # Initialize an empty sentence
        sentence_delay = 5  # Set the sentence delay in seconds
        last_recognition_time = time.time()  # Initialize last recognition time

        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            # Remove markings from the face
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            keypoints = extract_keypoints(results)

            sequence.insert(0, keypoints)  # Insert the new frame into the sequence
            sequence = sequence[:30]  # Keep only the last 30 frames

            # Recognize sign language
            if len(sequence) == 30:
                sequence_array = np.array(sequence)
                res = sign_model.predict(np.expand_dims(sequence_array, axis=0))
                recognized_action = actions[np.argmax(res[0])]

                cv2.putText(image, f"Recognized Action: {recognized_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if time.time() - last_recognition_time >= sentence_delay:
                    recognized_sentence += recognized_action + " "  # Add the recognized word to the sentence
                    last_recognition_time = time.time()

                # Recognize emotion
                emotion_labels = recognize_emotion(frame, face_classifier, emotion_model, ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
                
                # Check if emotion_labels is not empty before finding the dominant emotion
                if emotion_labels:
                    dominant_emotion = max(set(emotion_labels), key=emotion_labels.count)
                    cv2.putText(image, f"Dominant Emotion: {dominant_emotion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "No Faces Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Combine recognized sentence and emotion
                complete_sentence = recognized_sentence.strip() + f" [{dominant_emotion}]"

                cv2.putText(image, f"Complete Sentence: {complete_sentence}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return complete_sentence

# Streamlit app

def main():
    st.title("VocAI")
    st.text("Press 'Start' to begin capturing sign language gestures.")
    
    if st.button("Start"):
        st.text("Capturing... Press 'Stop' when done.")
        recognized_sentence = integrate_emotion_and_sign_language()
        
        st.text("Recognition complete. The recognized sentence is:")
        st.text(recognized_sentence)

        # Generate audio with the recognized sentence
        audio_text = generate_audio(recognized_sentence)

        # Check if audio generation was successful
        if audio_text is not None:
            st.audio(audio_text, format="audio/wav", start_time=0)
        else:
            st.warning("Error in audio generation.")

if __name__ == "__main__":
    actions = ['hello', 'thanks', 'iloveyou', 'beautiful', 'happy', 'loud']
    main()