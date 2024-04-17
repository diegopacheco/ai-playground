import os
import requests

# Face Recognition
import face_recognition

# Image Manipulation
import imageio.v2 as imageio
from io import BytesIO

# Model Deploy
import gradio as gr

raw_players_path = "./data/raw/players/"

def player_recogn(url_player_img):
    current_list_players = os.listdir(raw_players_path)
    scores = {player: 1.0 for player in current_list_players}
    
    # Headers for scrapping
    headers = {
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
    }

    # Searching the image in the internet
    answer_player_img_req = requests.get(url_player_img, headers=headers)
    new_img = imageio.imread(BytesIO(answer_player_img_req.content))

    # For each face in the image, returns the code with 128 dimensions
    new_img_coding = face_recognition.face_encodings(new_img)

    # If no face was found, returns error message
    if len(new_img_coding) == 0:
        print("Coudn't find any faces!")
        return None
    
    face_code = new_img_coding[0]
    face_code = [face_code]

    # Compare with all faces in the database and check the distance between the current detection to each of the images stored
    for player in current_list_players:
        # Full directory path
        full_path = os.path.join(raw_players_path, player)
        
        # List of images in that directory
        img_list = os.listdir(full_path)

        # If no image is in the directory, pass
        if len(img_list) == 0:
            continue

        # Setting initial distance to 0
        player_face_score = 0

        # Encoding the image in the database that is been used to compare
        for img in img_list:
            # Path of the image that is been used to recognize
            img_path = os.path.join(raw_players_path, player, img)

            # Turning database image in a numpy array
            player_img = imageio.imread(img_path)

            # Encoding database image
            player_img_code = face_recognition.face_encodings(player_img)

            # If the image can't be encoded, pass
            if len(player_img_code) == 0:
                continue
            else:
                player_face_code = player_img_code[0]

                # Compare the distance between the images
                distance_faces = face_recognition.face_distance(face_code, player_face_code)
                player_face_score += distance_faces[0]

        # Dictionary of scores for each image in each player)
        scores[player] = player_face_score / len(img_list)
    
    print(scores)

    # Return the minimum value since it is the smallest distance between images
    return min(scores, key=scores.get)

interface_gradio = gr.Interface(fn = player_recogn,
                                inputs = "textbox",
                                outputs = "textbox")

interface_gradio.launch(share=True)
