from deepface import DeepFace

result = DeepFace.verify(
  img1_path = "pic1.jpg",
  img2_path = "pic2.jpg",
)
print("Is verified: ", result)