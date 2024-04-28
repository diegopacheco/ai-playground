import requests
import os

response = requests.post(
    f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
    headers={
        "authorization": f"Bearer " + str(os.environ.get("STABILITY_TOKEN")),
        "accept": "image/*"
    },
    files={"none": ''},
    data={
        "prompt": "cat wearing black glasses",
        "output_format": "jpeg",
    },
)

if response.status_code == 200:
    with open("./cat.jpeg", 'wb') as file:
        file.write(response.content)
else:
    raise Exception(str(response.json()))

print("Image saved as cat.jpeg")