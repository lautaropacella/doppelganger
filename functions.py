import numpy as np
from deepface import DeepFace
import streamlit as st
import random

MALE_ICONS = [':man:', 
':man_in_tuxedo:',
':male-elf:', 
':male-scientist:', 
':male-cook:', 
':male-student:',
':male-singer:',
':male-artist:',
':male-teacher:',
':male-technologist:',
':male-office-worker:',
':male-pilot:',
':male-doctor:',
':male-astronaut:',
':male-mechanic:']
FEMALE_ICONS = [':woman:',
':woman-in-tuxedo:', 
':female-elf:', 
':female-scientist:', 
':female-cook:', 
':female-student:',
':female-singer:',
':female-artist:',
':female-teacher:',
':female-technologist:',
':female-office-worker:',
':female-pilot:',
':female-doctor:',
':female-astronaut:',
':female-mechanic:']

ETNIAS = {'asian':'Asiática', 
'indian':'HIndú', 
'black':'Africana', 
'white':'Caucásica', 
'middle eastern':'Medio-Oriente', 
'latino hispanic':'Latino/Hispano'}

def crop_face(img,left,top,right,bottom):

    scale_factor = 0.8

    ## Calculate center points and rectangle side length
    width = right - left
    height = bottom - top
    cX = left + width // 2
    cY = top + height // 2
    M = (abs(width) + abs(height)) / 2

    ## Get the resized rectangle points
    newLeft = max(0, int(cX - scale_factor * M))
    newTop = max(0, int(cY - scale_factor * M))
    newRight = min(img.shape[1], int(cX + scale_factor * M))
    newBottom = min(img.shape[0], int(cY + scale_factor * M))

    ## Draw the circle and bounding boxes
    face_crop = np.copy(img[newTop:newBottom, newLeft:newRight])
    return face_crop

@st.cache(allow_output_mutation=True)
def face_features(face):
    emotions_icon = {'happy': ':smile:', 'sad': ':cry:', 'angry': ':angry:', 'fear': ':fearful:', 'disgust': ':confounded:', 'neutral': ':neutral_face:'}
    obj = DeepFace.analyze(face ,actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False)
    age = obj.get('age')
    gender = obj.get('gender')
    race_eng = obj.get('dominant_race')
    emotion = obj.get('dominant_emotion')
    if gender == 'Man':
        gender_icon = random.choice(MALE_ICONS)
    else:
        gender_icon = random.choice(FEMALE_ICONS)
    emotion_icon = emotions_icon.get(emotion)
    race = ETNIAS.get(race_eng)
    
    return age,gender_icon,race,emotion_icon