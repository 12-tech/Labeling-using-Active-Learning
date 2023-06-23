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


datapath = Path(os.getcwd())/"data/Sugarbeet"
print(datapath)


# In[4]:


Path(os.getcwd())/"data/Sugarbeet"


# In[5]:


Path(os.getcwd())/"Sugarbeet/stage_1"
Path(os.getcwd())/"Sugarbeet/stage_2"


# In[6]:


import os

data_path = os.path.join(os.getcwd(), "Sugarbeet")

apple_healthy_path = os.path.join(data_path, "stage_1")
blueberry_healthy_path = os.path.join(data_path, "stage_2")

print("Files in stage_1 directory:")
print(os.listdir(apple_healthy_path))

print("Files in stage_2 directory:")
print(os.listdir(blueberry_healthy_path))


# In[7]:


import os
from PIL import Image

image_path = "Sugarbeet"
apple_healthy_path = os.path.join(image_path, "stage_1")
blueberry_healthy_path = os.path.join(image_path, "stage_2")

# Loop through all the files in the directory
for filename in os.listdir(apple_healthy_path):
    # Check if the file is an image file
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image file
        apple_img = Image.open(os.path.join(apple_healthy_path, filename))
        # Do something with the image
        print(apple_img.size)


# In[8]:


from fastai.vision.all import *

# Define path to image dataset
path = Path("Sugarbeet")

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


# In[9]:


dls = data.dataloaders(Path("Sugarbeet"))
dls.show_batch()


# In[10]:


# Load a pre-trained model
learn = vision_learner(dls, resnet50, metrics=accuracy)


# In[11]:


# Fine-tune the model
learn.fine_tune(1)


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


# In[18]:


import os

data_path = os.path.join(os.getcwd(), "unlabeled")

image_files = [file for file in os.listdir(data_path) if file.endswith(".jpg")]

print("Files in unlabeled directory:")
print(image_files)


# In[20]:


import os
from PIL import Image

image_path = "unlabeled"

# Loop through all the files in the directory
for filename in os.listdir(image_path):
    # Check if the file is an image file
    if filename.endswith(".jpg"):
        # Open the image file
        image_file = os.path.join(image_path, filename)
        with Image.open(image_file) as img:
            # Do something with the image
            print(img.size)


# In[21]:


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


# In[25]:


import torch

# Assuming you have a trained model named 'model'
model = resnet50(pretrained=False)

# Specify the file path where you want to save the model
model_path = 'model.newpth'

# Save the model
torch.save(model.state_dict(), model_path)


# In[27]:


model.load_state_dict(torch.load('model.newpth'))


# In[28]:


from torchvision.models import resnet50

model = resnet50()
print(model)


# In[35]:


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

# Iterate over the images in the folder
for image_file in os.listdir(unlabeled_data_path):
    # Construct the path to the image file
    image_path = os.path.join(unlabeled_data_path, image_file)
    
    # Check if the path is a file
    if not os.path.isfile(image_path):
        continue
    
    # Load and preprocess the image
    img = Image.open(image_path)
    img = transform(img)
    
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
        uncertain_samples.append(image_path)

# Print the uncertain samples
for sample in uncertain_samples:
    print(sample)


# In[36]:


# Change the path to labeled data
labeled_data_path = "Sugarbeet"  # Path to the directory containing labeled images

# Specify the maximum number of samples to label
max_samples_to_label = 50

# Initialize a counter for labeled samples
labeled_samples_count = 0

# Iterate over the uncertain samples
for sample_path in uncertain_samples:
    # Check if the maximum number of samples to label has been reached
    if labeled_samples_count >= max_samples_to_label:
        break

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
    
    # Increment the labeled samples count
    labeled_samples_count += 1

print("Annotation completed!")


# In[37]:


import os

data_path = os.path.join(os.getcwd(), "Sugarbeet")

apple_healthy_path = os.path.join(data_path, "stage_1")
blueberry_healthy_path = os.path.join(data_path, "stage_2")

print("Files in stage_1 directory:")
print(os.listdir(apple_healthy_path))

print("Files in stage_2 directory:")
print(os.listdir(blueberry_healthy_path))


# In[38]:


import os
from PIL import Image

image_path = "Sugarbeet"
apple_healthy_path = os.path.join(image_path, "stage_1")
blueberry_healthy_path = os.path.join(image_path, "stage_2")

# Loop through all the files in the directory
for filename in os.listdir(apple_healthy_path):
    # Check if the file is an image file
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image file
        apple_img = Image.open(os.path.join(apple_healthy_path, filename))
        # Do something with the image
        print(apple_img.size)


# In[39]:


from fastai.vision.all import *

# Define path to image dataset
path = Path("Sugarbeet")

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


# In[40]:


dls = data.dataloaders(Path("Sugarbeet"))
dls.show_batch()


# In[41]:


# Set the model to training mode
model.train()

# Define your loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# In[44]:


import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# Set the labeled data path
labeled_data_path = "Sugarbeet"

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Create the labeled dataset
labeled_dataset = ImageFolder(labeled_data_path, transform=transform)

# Create a data loader for the labeled dataset
labeled_data_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)

# Train the model on the newly labeled data
num_epochs = 10  # Set the number of training epochs

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("------------------------")
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    
    for inputs, labels in labeled_data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
    
    train_loss = train_loss / len(labeled_dataset)
    train_accuracy = 100.0 * train_correct / len(labeled_dataset)
    
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")


# In[45]:


# Evaluate the model
acc = learn.validate()[1]
print(f"Accuracy: {acc}")


# In[ ]:




