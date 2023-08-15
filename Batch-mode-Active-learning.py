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


# In[9]:


dls = data.dataloaders(Path("Labeled_dataset"))
dls.show_batch()


# In[10]:


# Load a pre-trained model
learn = vision_learner(dls, resnet50, metrics=accuracy)


# In[11]:


# Fine-tune the model
learn.fine_tune(10)


# In[12]:


import matplotlib.pyplot as plt

# Get the learning rate values
lrs = learn.recorder.lrs

# Plot the learning rate schedule
plt.plot(range(len(lrs)), lrs)
plt.xlabel('Training Iterations')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()


# In[13]:


import matplotlib.pyplot as plt

# Create a recorder object
recorder = learn.recorder

# Plot the validation and training losses
recorder.plot_loss()


# In[14]:


# Evaluate the model
acc = learn.validate()[1]
print(f"Accuracy: {acc}")


# In[15]:


Path(os.getcwd())/"unlabeled"


# In[16]:


Path(os.getcwd())/"unlabeled"


# In[17]:


import os

data_path = os.path.join(os.getcwd(), "unlabeled")

image_files = [file for file in os.listdir(data_path) if file.lower().endswith((".jpg", ".jpeg"))]

print("Files in unlabeled directory:")
print(image_files)


# In[18]:


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


# In[19]:


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


# In[20]:


#Select a batch of unlabeled samples
batch_size = 100  # Set the desired batch size
unlabeled_samples = unlabeled_dls.train.items[:batch_size]


# In[21]:


# Print the selected unlabeled samples
print(unlabeled_samples)

# Check the number of samples
print(len(unlabeled_samples))


# In[22]:


# Specify the file path where you want to save the model
model_path = 'model.batch'

# Save the model
learn.save(model_path)


# In[23]:


learn.export('model.batch')


# In[24]:


learn = load_learner('model.batch')


# In[25]:


import os
from PIL import Image
from fastai.vision.all import *
from torchvision import transforms


# Path to the directory containing unlabeled images
unlabeled_data_path = "unlabeled"

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Define a threshold for confidence
confidence_threshold = 0.5

# Load the saved model using load_learner
model_path = 'model.batch'  
learn = load_learner(model_path)

# Access the model from the learner
model = learn.model

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# If your model uses custom functions like get_label, they are already defined in the learner
# No need to re-define them here

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
    
    # Move the preprocessed image to the same device as the model
    img = img.to(device)
    
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


# In[26]:


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


# In[27]:


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


# In[28]:


dls = data.dataloaders(Path("Labeled_dataset"))
dls.show_batch()


# In[29]:


from fastai.vision.all import *

learn = cnn_learner(dls, resnet50, metrics=accuracy)


# In[30]:


learn.load('model.batch')


# In[31]:


learn.fine_tune(10)  # Train for 10 epochs


# In[32]:


import matplotlib.pyplot as plt

# Get the training history from the learner
train_history = learn.recorder

# Create a list of epoch numbers (assuming each validation is done at the end of an epoch)
epochs = list(range(1, len(train_history.values) + 1))

# Plot the accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_history.values, label='Training Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[33]:


learn.save('new_batch_model')


# In[34]:


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


# In[35]:


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


# In[36]:


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


# In[37]:


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


# In[38]:


datapath = Path(os.getcwd())/"new_data"
print(datapath)


# In[39]:


Path(os.getcwd())/"new_data/Apple_healthy"
Path(os.getcwd())/"new_data/sugarbeet_stage_1"
Path(os.getcwd())/"new_data/sugarbeet_stage_2"


# In[40]:


import os
from PIL import Image

data_path = "new_data"
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


# In[41]:


import os
from PIL import Image
import torch
from torchvision import transforms
from fastai.learner import load_learner

# Load the Learner object with the trained model
model_path = 'model.batch' 
learn = load_learner(model_path)

# Set the model to evaluation mode
learn.model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Define the data directory
data_path = "new_data"  # Replace 'new_data' with the path to your data directory

# Define subdirectories within the data directory
sugarbeet_stage_1_path = os.path.join(data_path, "sugarbeet_stage_1")
sugarbeet_stage_2_path = os.path.join(data_path, "sugarbeet_stage_2")
apple_path = os.path.join(data_path, "Apple")

# Function to predict images in a given directory
def predict_images(directory_path):
    predictions = []
    for image_file in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_file)
        if not os.path.isfile(image_path):
            continue
        
        img = Image.open(image_path)
        img = transform(img)
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            prediction = learn.model(img)
        
        probabilities = torch.softmax(prediction, dim=1)
        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class_label = learn.dls.vocab[predicted_class_index]
        predictions.append(predicted_class_label)
    
    return predictions

# Predict images in each subdirectory
sugarbeet_stage_1_predictions = predict_images(sugarbeet_stage_1_path)
sugarbeet_stage_2_predictions = predict_images(sugarbeet_stage_2_path)
apple_predictions = predict_images(apple_path)

# Print the predictions
print("Sugarbeet Stage 1 Predictions:", sugarbeet_stage_1_predictions)
print("Sugarbeet Stage 2 Predictions:", sugarbeet_stage_2_predictions)
print("Apple Predictions:", apple_predictions)


# In[47]:


get_ipython().system('pip install seaborn numpy pandas matplotlib')
get_ipython().system('pip install --upgrade seaborn')


# In[51]:


import os
import numpy as np
import torch
from torchvision import transforms
from fastai.learner import load_learner
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the Learner object with the trained model
model_path = 'model.batch'
learn = load_learner(model_path)

# Set the model to evaluation mode
learn.model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Function to predict images in a given directory
def predict_images(directory_path):
    predictions = []
    for image_file in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_file)
        if not os.path.isfile(image_path):
            continue

        img = Image.open(image_path)
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            prediction = learn.model(img)

        probabilities = torch.softmax(prediction, dim=1)
        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class_label = learn.dls.vocab[predicted_class_index]
        predictions.append(predicted_class_label)

    return predictions

# Define true labels and predicted labels for each subdirectory
true_labels = []
sugarbeet_stage_1_predictions = predict_images(os.path.join(data_path, "sugarbeet_stage_1"))
sugarbeet_stage_2_predictions = predict_images(os.path.join(data_path, "sugarbeet_stage_2"))
apple_predictions = predict_images(os.path.join(data_path, "Apple"))

# Replace these lists with the true labels for each subdirectory
true_labels.extend(['sugarbeet_stage_1'] * len(sugarbeet_stage_1_predictions))
true_labels.extend(['sugarbeet_stage_2'] * len(sugarbeet_stage_2_predictions))
true_labels.extend(['Apple'] * len(apple_predictions))

# Combine the predicted labels from all subdirectories
all_predictions = sugarbeet_stage_1_predictions + sugarbeet_stage_2_predictions + apple_predictions

# Create the confusion matrix
conf_matrix = confusion_matrix(true_labels, all_predictions, labels=learn.dls.vocab)

print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

class_names = learn.dls.vocab
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

fmt = 'd'
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 ha='center', va='center',
                 color='white' if conf_matrix[i, j] > thresh else 'black')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()


# In[53]:


import os
import numpy as np
import torch
from torchvision import transforms
from fastai.learner import load_learner
from PIL import Image

# Load the Learner object with the trained model
model_path = 'model.batch'
learn = load_learner(model_path)

# Set the model to evaluation mode
learn.model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Function to predict images in a given directory and display wrongly predicted images
def predict_images(directory_path):
    wrongly_predicted_images = []
    for image_file in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_file)
        if not os.path.isfile(image_path):
            continue

        img = Image.open(image_path)
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            prediction = learn.model(img)

        probabilities = torch.softmax(prediction, dim=1)
        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class_label = learn.dls.vocab[predicted_class_index]

        true_label = os.path.basename(directory_path)

        if predicted_class_label != true_label:
            wrongly_predicted_images.append((image_path, true_label, predicted_class_label))

    return wrongly_predicted_images

# Define the data directory
data_path = "new_data"  # Replace 'new_data' with the path to your data directory

# Predict and display wrongly predicted images for each subdirectory
sugarbeet_stage_1_wrong = predict_images(os.path.join(data_path, "sugarbeet_stage_1"))
sugarbeet_stage_2_wrong = predict_images(os.path.join(data_path, "sugarbeet_stage_2"))
apple_wrong = predict_images(os.path.join(data_path, "Apple"))

# Display the wrongly predicted images
for image_path, true_label, predicted_label in sugarbeet_stage_1_wrong + sugarbeet_stage_2_wrong + apple_wrong:
    print("Image Path:", image_path)
    print("True Label:", true_label)
    print("Predicted Label:", predicted_label)
    print("")


# In[54]:


import os
import numpy as np
import torch
from torchvision import transforms
from fastai.learner import load_learner
from PIL import Image
import matplotlib.pyplot as plt

# Load the Learner object with the trained model
model_path = 'model.batch'
learn = load_learner(model_path)

# Set the model to evaluation mode
learn.model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Function to predict images in a given directory and get wrongly predicted images
def predict_images(directory_path):
    wrongly_predicted_images = []
    for image_file in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_file)
        if not os.path.isfile(image_path):
            continue

        img = Image.open(image_path)
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            prediction = learn.model(img)

        probabilities = torch.softmax(prediction, dim=1)
        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class_label = learn.dls.vocab[predicted_class_index]

        true_label = os.path.basename(directory_path)

        if predicted_class_label != true_label:
            wrongly_predicted_images.append((img, true_label, predicted_class_label))

    return wrongly_predicted_images

# Define the data directory
data_path = "new_data"  # Replace 'new_data' with the path to your data directory

# Predict and get wrongly predicted images for each subdirectory
sugarbeet_stage_1_wrong = predict_images(os.path.join(data_path, "sugarbeet_stage_1"))
sugarbeet_stage_2_wrong = predict_images(os.path.join(data_path, "sugarbeet_stage_2"))
apple_wrong = predict_images(os.path.join(data_path, "Apple"))

# Display a few wrongly predicted images
num_images_to_display = 3

for img, true_label, predicted_label in sugarbeet_stage_1_wrong[:num_images_to_display] + \
                                     sugarbeet_stage_2_wrong[:num_images_to_display] + \
                                     apple_wrong[:num_images_to_display]:
    # Convert the tensor image to a NumPy array
    img_np = img.squeeze(0).permute(1, 2, 0).numpy()
    
    # De-normalize the image
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    # Display the image
    plt.imshow(img_np)
    plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show()


# In[ ]:




