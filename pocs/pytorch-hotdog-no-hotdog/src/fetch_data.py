import os
import tarfile
import urllib.request

# Download the dataset
url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
filename = "food-101.tar.gz"
urllib.request.urlretrieve(url, filename)

# Extract the dataset
tar = tarfile.open(filename, "r:gz")
tar.extractall()
tar.close()

# Move the hotdog images to a new directory
os.rename("food-101/images/hot_dog", "hotdog")

# Create a 'not_hotdog' directory and move some other food images there
os.mkdir("not_hotdog")
for food in os.listdir("food-101/images"):
    if food != "hot_dog":
        for image in os.listdir(f"food-101/images/{food}")[:100]:  # get 100 images of each food
            os.rename(f"food-101/images/{food}/{image}", f"not_hotdog/{image}")

torch.save(model.state_dict(), 'hotdog_not_hotdog.pth')
print("Model saved!")