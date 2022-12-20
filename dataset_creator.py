# Covid X-Ray Image Dataset - https://github.com/ieee8023/covid-che... for positive cases.

# Kaggle X-Ray Chest Images - https://www.kaggle.com/paultimothymoo... 
# for negative cases.

import pandas as pd
import os
import shutil
file_path = "metadata.csv"
images_path = "images"

df = pd.read_csv(file_path)

print(df.shape)
df.head()

target = "dataset/train/covid"

if not os.path.exists(target):
    os.mkdir(target)
    print("Creating covid folder")

cnt=0
for(i,row) in df.iterrows():
    if row["finding"]=="Pneumonia/Viral/COVID-19" and row["view"]=="PA":
        filename = row["filename"]
        image_path = os.path.join(images_path,filename)
        image_copy_path = os.path.join(target,filename)
        shutil.copy2(image_path,image_copy_path)
        #print("moving image ",cnt)
        cnt+=1

import random
kaggle_path="normal"
target_dir = "dataset/train/normal"

image_names = os.listdir(kaggle_path)
random.shuffle(image_names)

for i in range(1341):
    image_name = image_names[i]
    image_path = os.path.join(kaggle_path,image_name)
    target_path = os.path.join(target_dir,image_name)
    shutil.copy2(image_path,target_path)
    print("Copying ",i)
