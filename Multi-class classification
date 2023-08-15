#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.data.all import *
from fastai.vision.all import *
from fastai.vision.all import *
from pathlib import Path
import torch
torch.cuda.set_device(3)


# In[2]:


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# In[3]:


datapath = Path(os.getcwd())/"Labeled_dataset"
print(datapath)


# In[4]:


Path(os.getcwd())/"Labeled_dataset"


# In[5]:


Path(os.getcwd())/"Labeled_dataset/sugarbeet_stage_1"
Path(os.getcwd())/"Labeled_dataset/sugarbeet_stage_2"
Path(os.getcwd())/"Labeled_dataset/Apple"


# In[6]:


import os
from PIL import Image

data_path = "Labeled_dataset"
sugarbeet_stage_1_path = os.path.join(data_path, "sugarbeet_stage_1")
sugarbeet_stage_2_path = os.path.join(data_path, "sugarbeet_stage_2")
apple_path = os.path.join(data_path, "Apple")

# Loop through all the files in the Sugarbeet-Stage1 directory
print("Files in sugarbeet_stage_1 directory:")
for filename in os.listdir(sugarbeet_stage_1_path):
    if filename.endswith(".jpg"):
        print(filename)

# Loop through all the files in the Sugarbeet-Stage2 directory
print("Files in sugarbeet_stage_2 directory:")
for filename in os.listdir(sugarbeet_stage_2_path):
    if filename.endswith(".jpg"):
        print(filename)

# Loop through all the files in the Apple directory
print("Files in Apple directory:")
for filename in os.listdir(apple_path):
    if filename.endswith(".JPG"):
        print(filename)


# In[7]:


import os
from PIL import Image

image_path = "Labeled_dataset"
sugarbeet_stage_1_path = os.path.join(image_path, "sugarbeet_stage_1")
sugarbeet_stage_2_path = os.path.join(image_path, "sugarbeet_stage_2")
Apple_path = os.path.join(image_path, "Apple")

# Loop through all the files in the directory
for filename in os.listdir(Apple_path):
    # Check if the file is an image file
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
        # Open the image file
        Apple_img = Image.open(os.path.join(Apple_path, filename))
        # Do something with the image
        print(Apple_img.size)


# In[8]:


from fastai.vision.all import *

# Define path to image dataset
path = Path("Labeled_dataset")

# Define a function to get the label of an image based on its path
def get_label(file_path):
    return file_path.parent.name

# Define a DataBlock for loading and transforming the images
data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(),
    get_y=get_label,
    item_tfms=[Resize(224)],
    batch_tfms=[*aug_transforms()]
)

# Create dataloaders from the data
dls = data.dataloaders(path)


# In[10]:


dls = data.dataloaders(Path("Labeled_dataset"))
dls.show_batch()


# In[11]:


# Load a pre-trained model
learn = vision_learner(dls, resnet50, metrics=accuracy)


# In[12]:


# Fine-tune the model
learn.fine_tune(10)


# In[13]:


import matplotlib.pyplot as plt

# Get the learning rate values
lrs = learn.recorder.lrs

# Plot the learning rate schedule
plt.plot(range(len(lrs)), lrs)
plt.xlabel('Training Iterations')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()


# In[14]:


import matplotlib.pyplot as plt

# Create a recorder object
recorder = learn.recorder

# Plot the validation and training losses
recorder.plot_loss()


# In[15]:


# Evaluate the model
acc = learn.validate()[1]
print(f"Accuracy: {acc}")


# In[16]:


Path(os.getcwd())/"unlabeled"


# In[17]:


Path(os.getcwd())/"unlabeled"


# In[18]:


import os

data_path = os.path.join(os.getcwd(), "unlabeled")

image_files = [file for file in os.listdir(data_path) if file.lower().endswith((".jpg", ".jpeg"))]

print("Files in unlabeled directory:")
print(image_files)


# In[19]:


import os
from PIL import Image

image_path = "unlabeled"

# Loop through all the files in the directory
for filename in os.listdir(image_path):
    # Check if the file is an image file
    if filename.endswith((".jpg", ".JPG")):
        # Open the image file
        image_file = os.path.join(image_path, filename)
        with Image.open(image_file) as img:
            # Do something with the image
            print(img.size)


# In[20]:


from fastai.vision.all import *

unlabeled_path = Path("unlabeled")

# Define a DataBlock for loading the unlabeled images
unlabeled_data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,

    get_y=noop,  # No labels for unlabeled data
    item_tfms=[Resize(224)]
)

# Create a dataloader for the unlabeled data
unlabeled_dls = unlabeled_data.dataloaders(unlabeled_path, bs=64)


# In[22]:


#Select a batch of unlabeled samples
batch_size = 100  # Set the desired batch size
unlabeled_samples = unlabeled_dls.train.items[:batch_size]


# In[30]:


# Print the selected unlabeled samples
print(unlabeled_samples)

# Check the number of samples
print(len(unlabeled_samples))


# In[36]:


# Specify the file path where you want to save the model
model_path = 'model.batch'

# Save the model
learn.save(model_path)


# In[39]:


learn.load(model_path)


# In[50]:


import os
from PIL import Image
import torch
from torchvision import transforms

unlabeled_data_path = "unlabeled"  # Path to the directory containing unlabeled images

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Define a threshold for confidence
confidence_threshold = 0.5

# Initialize a list to store the uncertain samples
uncertain_samples = []

# Iterate over the files in the folder
for file_name in os.listdir(unlabeled_data_path):
    # Construct the path to the file
    file_path = os.path.join(unlabeled_data_path, file_name)
    
    # Check if the path is a file and has a valid image file extension
    if not os.path.isfile(file_path) or not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue
    
    # Load and preprocess the image
    try:
        img = Image.open(file_path)
        img = transform(img)
    except Exception as e:
        print(f"Error opening image file: {file_path}")
        continue
    
    # Add an extra dimension to match the batch size expected by the model
    img = img.unsqueeze(0)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient calculation since we're only doing inference
    with torch.no_grad():
        # Pass the preprocessed image through the model
        prediction = model(img)
    
    # Convert the prediction to probabilities using softmax
    probabilities = torch.softmax(prediction, dim=1)
    
    # Get the maximum predicted probability and its corresponding class
    max_prob, predicted_class = torch.max(probabilities, dim=1)
    
    # Check if the maximum predicted probability is below the confidence threshold
    if max_prob.item() < confidence_threshold:
        uncertain_samples.append(file_path)

# Print the uncertain samples
for sample in uncertain_samples:
    print(sample)


# In[51]:


labeled_data_path = "Labeled_dataset"  # Path to the directory containing labeled images

# Iterate over the uncertain samples
for sample_path in uncertain_samples:
    # Ask the user for the correct label
    correct_label = input(f"Enter the correct label for sample {sample_path}: ")
    
    # Create the subfolder path for the correct label
    correct_label_path = os.path.join(labeled_data_path, correct_label)
    
    # Create the subfolder if it doesn't exist
    os.makedirs(correct_label_path, exist_ok=True)
    
    # Move the sample to the correct label subfolder
    sample_filename = os.path.basename(sample_path)
    new_sample_path = os.path.join(correct_label_path, sample_filename)
    os.rename(sample_path, new_sample_path)

print("Annotation completed!")


# In[53]:


from fastai.vision.all import *

# Define path to image dataset
path = Path("Labeled_dataset")

# Define a function to get the label of an image based on its path
def get_label(file_path):
    return file_path.parent.name

# Define a DataBlock for loading and transforming the images
data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(),
    get_y=get_label,
    item_tfms=[Resize(224)],
    batch_tfms=[*aug_transforms()]
)

# Create dataloaders from the data
dls = data.dataloaders(path)


# In[57]:


dls = data.dataloaders(Path("Labeled_dataset"))
dls.show_batch()


# In[65]:


from fastai.vision.all import *

learn = cnn_learner(dls, resnet50, metrics=accuracy)


# In[67]:


learn.load('model.batch')


# In[68]:


learn.fine_tune(10)  # Train for 10 epochs


# In[69]:


valid_loss, valid_acc = learn.validate()
print(f"Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}")


# In[71]:


learn.save('retrained_model')


# In[82]:


import PIL

# Define the path to your new image
new_image_path = "unlabeled/14e92721-00a0-4696-9392-d3f633082454___RS_HL 5654.jpg"

# Load the image using the PIL library
img = PIL.Image.open(new_image_path)

# Show the image
plt.imshow(img)
plt.axis('off')
plt.show()

# Make a prediction using the trained model
pred, _, _ = learn.predict(new_image_path)

# Print the predicted label
print(f"Prediction: {pred}")


# In[83]:


import PIL

# Define the path to your new image
new_image_path = "unlabeled/dji-55-image-00612.jpg"

# Load the image using the PIL library
img = PIL.Image.open(new_image_path)

# Show the image
plt.imshow(img)
plt.axis('off')
plt.show()

# Make a prediction using the trained model
pred, _, _ = learn.predict(new_image_path)

# Print the predicted label
print(f"Prediction: {pred}")


# In[87]:


import os
import PIL

# Define the path to your unlabeled data directory
unlabeled_data_dir = 'unlabeled'

# Get a list of image files in the unlabeled data directory, excluding the .ipynb_checkpoints directory
image_files = [file_name for file_name in os.listdir(unlabeled_data_dir) if not file_name.startswith('.ipynb_checkpoints')]

# Select the first 10 images for prediction
image_files = image_files[:10]

# Iterate over the image files
for file_name in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(unlabeled_data_dir, file_name)

    # Load the image using the PIL library
    img = PIL.Image.open(image_path)

    # Show the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Make a prediction using the trained model
    pred, _, _ = learn.predict(image_path)

    # Print the predicted label
    print(f"Prediction for {file_name}: {pred}")


# In[88]:


import os
import PIL

# Define the path to your unlabeled data directory
unlabeled_data_dir = 'unlabeled'

# Get a list of image files in the unlabeled data directory, excluding the .ipynb_checkpoints directory
image_files = [file_name for file_name in os.listdir(unlabeled_data_dir) if not file_name.startswith('.ipynb_checkpoints')]

# Select the last 10 images for prediction
image_files = image_files[-10:]

# Iterate over the image files
for file_name in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(unlabeled_data_dir, file_name)

    # Load the image using the PIL library
    img = PIL.Image.open(image_path)

    # Show the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Make a prediction using the trained model
    pred, _, _ = learn.predict(image_path)

    # Print the predicted label
    print(f"Prediction for {file_name}: {pred}")


# In[ ]:




