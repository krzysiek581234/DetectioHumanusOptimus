import cv2
import torch
from torchvision import transforms
from net import CNN_NET
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import filedialog






def Sliding_Window():
    image = original_image
    model = CNN_NET()
    model.load_state_dict(torch.load('CNN.pth'))
    model.eval()
    stride = 2  # Stride for moving the window
    # Initialize lists to store detected objects and their locations
    detected_objects = []
    object_locations = []
    size_of_windows = [36]
    for s in size_of_windows:
        window_size = (s,s)
        for y in range(0, image.shape[0], stride):
            for x in range(0, image.shape[1], stride):
                # print(f" x: {x}, y:  {y}")
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
                    object_locations.append((x, y, x + window_size[0]/2, y + window_size[1]/2, window_size[0]))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    new_object_locations = []  # Create a new list for updated object locations

    for i, (x1, y1, xc1, yc1, window_size1) in enumerate(object_locations):
        found_match = False  # Flag to check if an object was matched
        for j, (x2, y2, xc2, yc2, window_size2) in enumerate(object_locations):
            if i != j:  # Skip comparing an object with itself
                distance_x = (xc1 - xc2)
                distance_y = (yc1 - yc2)
                distance = math.sqrt(distance_x**2 + distance_y**2)
                if distance < abs(window_size1 - window_size2 - 6):
                    new_x = (x1 + x2) / 2
                    new_y = (y1 + y2) / 2
                    new_window_size = window_size1 + window_size2 - 6
                    new_object_locations.append((new_x, new_y, new_x + new_window_size/2, new_y + new_window_size/2, new_window_size))
                    found_match = True

        if not found_match:
            # If no match was found, keep the object as is
            new_object_locations.append((x1, y1, xc1, yc1, window_size1))

    # Update the object_locations with the new list
    object_locations = new_object_locations

    for (x1, y1, x2, y2,w) in object_locations:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=(0, 1, 0), facecolor="none")
        plt.gca().add_patch(rect)
    plt.title("Detected Objects")
    plt.show()

        





def choose_image():
    global selected_image, original_image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # load the chosen image and display it on the window
    image = cv2.imread(file_path)
    selected_image = image.copy()
    original_image = image.copy()

    desired_width = 800
    aspect_ratio = desired_width / image.shape[1]
    new_height = int(image.shape[0] * aspect_ratio)

    image = cv2.resize(image, (desired_width, new_height))
    chosen_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chosen_image = Image.fromarray(chosen_image)
    chosen_image_tk = ImageTk.PhotoImage(chosen_image)
    display_image_label.configure(image=chosen_image_tk)
    display_image_label.image = chosen_image_tk

    # update the window geometry based on the image size
    window_width = chosen_image.width + 150
    window_height = chosen_image.height + 20
    window.geometry(f"{window_width}x{window_height}")

if __name__ == '__main__':
    window = tk.Tk()
    window.title("Emotion Detection")

    window.geometry("960x540")
    window.resizable(True, True)

    buttons_frame = tk.Frame(window)
    buttons_frame.pack(side="right", padx=10)

    file_button = tk.Button(buttons_frame, text="Choose Image", command=choose_image)
    file_button.pack(pady=10)

    face_detection_button = tk.Button(buttons_frame, text="Selective Search", command=choose_image)
    face_detection_button.pack(pady=10)

    cnn_button = tk.Button(buttons_frame, text="Sliding Window", command=Sliding_Window)
    cnn_button.pack(pady=10)

    feedforward_button = tk.Button(buttons_frame, text="Single-Shot Multiple box Detector", command=choose_image)
    feedforward_button.pack(pady=10)

    labels_frame = tk.Frame(window)
    labels_frame.pack()

    display_image_label = tk.Label(window)
    display_image_label.pack(padx=10, pady=10)

    canvas = tk.Canvas(window, width=800, height=960)
    canvas.pack(padx=10, pady=10)
    window.mainloop()
        