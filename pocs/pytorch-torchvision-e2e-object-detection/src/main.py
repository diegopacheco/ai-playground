import pathlib
import torch
import torch.utils.data
from torchvision import models, datasets, tv_tensors
from torchvision.transforms import v2
import matplotlib.pyplot as plt

torch.manual_seed(0)

# This loads fake data for illustration purposes of this example. In practice, you'll have
# to replace this with the proper data.
# If you're trying to run that on collab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
ROOT = pathlib.Path("/home/diego/diego/data-ai/vision/gallery/assets") / "coco"
IMAGES_PATH = str(ROOT / "images")
ANNOTATIONS_PATH = str(ROOT / "instances.json")

##
## Data preparation
##
dataset = datasets.CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH)

sample = dataset[0]
img, target = sample
print(f"{type(img) = }\n{type(target) = }\n{type(target[0]) = }\n{target[0].keys() = }")

ataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels", "masks"))

sample = dataset[0]
img, target = sample
print(f"{type(img) = }\n{type(target) = }\n{target[0].keys() = }")

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomPhotometricDistort(p=1),
        v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=1),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

dataset = datasets.CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH, transforms=transforms)
dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels", "masks"])

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    # We need a custom collation function here, since the object detection
    # models expect a sequence of images and target dictionaries. The default
    # collation function tries to torch.stack() the individual elements,
    # which fails in general for object detection, because the number of bounding
    # boxes varies between the images of the same batch.
    collate_fn=lambda batch: tuple(zip(*batch)),
)

model = models.get_model("maskrcnn_resnet50_fpn_v2", weights=None, weights_backbone=None).train()

for imgs, targets in data_loader:
    loss_dict = model(imgs, targets)
    # Put your training logic here

    print(f"{[img.shape for img in imgs] = }")
    print(f"{[type(target) for target in targets] = }")
    for name, loss_val in loss_dict.items():
        print(f"{name:<20}{loss_val:.3f}")

## predict on a single image
def predict(model, img):
    model.eval()
    with torch.no_grad():
        # Unsqueeze the image to add an extra batch dimension
        img = img.unsqueeze(0)
        pred = model(img)
    return pred

img, target = dataset[0]
pred = predict(model, img)
print(f"prediction: {pred = }")


import matplotlib.pyplot as plt

# Plot the original image
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img.permute(1, 2, 0)) # permute the dimensions to match the expected input of plt.imshow
plt.title('Original Image')

# Plot the predicted mask
if pred[0]['masks'].numel() > 0: # Check if any masks were predicted
    plt.subplot(1,2,2)
    plt.imshow(pred[0]['masks'][0, 0].mul(255).byte().cpu().numpy(), cmap='gray')
    plt.title('Predicted Mask')
else:
    print("No masks predicted")

plt.show()

