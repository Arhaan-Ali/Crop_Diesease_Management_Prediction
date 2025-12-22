import os
import random
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "dataset"
classes = os.listdir(dataset_path)

samples_per_class = 5

plt.figure(figsize=(15, 10))

plot_index = 1

for disease in classes:
    disease_path = os.path.join(dataset_path, disease)
    images = os.listdir(disease_path)

    selected_images = random.sample(images, samples_per_class)

    for img_name in selected_images:
        img_path = os.path.join(disease_path, img_name)
        img = Image.open(img_path)

        plt.subplot(len(classes), samples_per_class, plot_index)
        plt.imshow(img)
        plt.axis("off")

        if plot_index % samples_per_class == 1:
            plt.ylabel(disease, fontsize=10)

        plot_index += 1

plt.tight_layout()
plt.show()
