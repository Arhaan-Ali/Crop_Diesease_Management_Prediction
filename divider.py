import os
import random
import shutil
images = []

for entry in os.scandir("ResizedData/Tungro"):
    images.append(entry.name)

Total = len(images)
Test = int(Total * (15 / 100))
Validate = Test
Train = Total - (2 * Test)

random.shuffle(images)

for i in range(0 ,Test):
    src = "ResizedData/Tungro/" + images[i]
    out = "Finalised_Dataset/Tungro/Tungro_Test/" + images[i]
    shutil.copy(src, out)

for i in range(Test, (Validate + Test)):
    src = "ResizedData/Tungro/" + images[i]
    out = "Finalised_Dataset/Tungro/Tungro_Validate/" + images[i]
    shutil.copy(src, out)

for i in range((Validate + Test), Total):
    src = "ResizedData/Tungro/" + images[i]
    out = "Finalised_Dataset/Tungro/Tungro_Train/" + images[i]
    shutil.copy(src, out)