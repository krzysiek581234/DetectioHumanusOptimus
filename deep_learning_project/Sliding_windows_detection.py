import cv2
import torch
from torchvision import transforms
from net import CNN_NET
from PIL import Image
import matplotlib.pyplot as plt
model = CNN_NET()
model.load_state_dict(torch.load('CNN.pth'))
model.eval()

window_size = (36, 36)  # Size of the sliding window
stride = 2  # Stride for moving the window

image = cv2.imread('rodzina.jpeg')

# Initialize lists to store detected objects and their locations
detected_objects = []
object_locations = []



for y in range(0, image.shape[0], stride):
    for x in range(0, image.shape[1], stride):
        print(f" x: {x}, y:  {y}")
        window = image[y:y+window_size[1], x:x+window_size[0]]
        window_pil = Image.fromarray(window)
        transform = transforms.Compose(
                [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
                transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
                transforms.Normalize(mean=(0.5,),std=(0.5,)),
                transforms.Resize((36, 36))
                ]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)
        window = transform(window_pil).unsqueeze(0)
        with torch.no_grad():
            output = model(window)
        _, predicted = torch.max(output.data, 1)
        if predicted == 1:
            print("JEST")
            object_locations.append((x, y, x + window_size[0], y + window_size[1]))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
for (x1, y1, x2, y2) in object_locations:
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=(0, 1, 0), facecolor="none")
    plt.gca().add_patch(rect)
plt.title("Detected Objects")
plt.show()



        