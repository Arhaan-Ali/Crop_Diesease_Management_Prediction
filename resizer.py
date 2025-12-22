from PIL import Image
import os

def resize_and_crop(img, size=224):
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    right = left + size
    bottom = top + size

    return img.crop((left, top, right, bottom))

input_dir = "dataset/Tungro"
output_dir = "ResizedData/Tungro"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(os.path.join(input_dir, file)).convert("RGB")
        img = resize_and_crop(img, 224)
        img.save(os.path.join(output_dir, file))

print("All images resized + cropped to 224x224")
