# Auto‑generated training / inference script extracted from notebook
import os
import cv2
import pandas as pd
import random
import numpy as np
from PIL import Image
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, GlobalAveragePooling2D, Flatten, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
from deepface import DeepFace
from transformers import CLIPProcessor, CLIPModel
# Paths
image_folder = 'WFLW_images'
output_folder = 'aligned_results'
annotation_file = 'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
test_annotation_file = 'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
augmented_folder = 'augmented_results'  
transformed_folder = 'transformed_data'
    """ 
    Load annotations from the text file. This function accesses a text file of facial landmarks 
    and attributes using the provided file path. Each line in the file contains data for one image, 
    including landmarks, attributes, and bounding box coordinates, which are processed and organized 
    into a DataFrame for further use.
    Args:
        file_path (str): Path to the annotations file.
    Returns:
        pd.DataFrame: DataFrame containing the annotations.
    """
    data = pd.read_csv(file_path, sep='\s+', header=None)  # Read data from the file assuming space-separated values.
    num_coords = 196  # Number of numerical entries dedicated to landmark coordinates (x and y for 98 points).
    num_rect_coords = 4  # Number of values representing the bounding box coordinates.
    # Extract image names and attribute columns based on predefined indices.
    data['image_name'] = data.iloc[:, num_coords + num_rect_coords + 6].values
    attributes_columns = num_coords + num_rect_coords + np.arange(6)  # Indices for six attribute columns.
    data['pose'] = data.iloc[:, attributes_columns[0]].values
    data['expression'] = data.iloc[:, attributes_columns[1]].values
    data['illumination'] = data.iloc[:, attributes_columns[2]].values
    data['make-up'] = data.iloc[:, attributes_columns[3]].values
    data['occlusion'] = data.iloc[:, attributes_columns[4]].values
    data['blur'] = data.iloc[:, attributes_columns[5]].values
    # Extract bounding box coordinates based on predefined indices.
    rect_columns = num_coords + np.arange(num_rect_coords)  # Indices for bounding box coordinates.
    data['x_min_rect'] = data.iloc[:, rect_columns[0]].values
    data['y_min_rect'] = data.iloc[:, rect_columns[1]].values
    data['x_max_rect'] = data.iloc[:, rect_columns[2]].values
    data['y_max_rect'] = data.iloc[:, rect_columns[3]].values
    return data
    """ 
    Preprocess images by cropping faces based on bounding box coordinates and saving them.
    Args:
        data (pd.DataFrame): DataFrame containing image names and bounding box coordinates.
        image_folder (str): Path to the folder containing the original images.
        output_folder (str): Path to the folder where processed images will be saved.
    """
    for idx, row in data.iterrows():
        image_path = os.path.join(image_folder, row['image_name'])
        image = cv2.imread(image_path)
        if image is not None:
            x_min = row['x_min_rect']
            y_min = row['y_min_rect']
            x_max = row['x_max_rect']
            y_max = row['y_max_rect']
            face = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            output_path = os.path.join(output_folder, row['image_name'].split('/')[-1])
            if face.size > 0:
                face_RGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB.
                img = Image.fromarray(face_RGB)  # Convert the array into an image object.
                img.save(output_path)  # Save the image object to the specified directory.
                print(f"Cropped face saved to {output_path}")
        else:
            print(f"Image not found: {image_path}")
    """ 
    Split the data into training and validation sets based on the specified test size proportion.
    Args:
        data (pd.DataFrame): DataFrame containing the annotations and attributes.
        test_size (float): Proportion of the data to be used as validation set.
    Returns:
        tuple: Training and validation DataFrames.
    """
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, val_data
# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Load and preprocess training and validation data
data = load_annotations(annotation_file)
train_data, val_data = split_data(data)
preprocess_images(train_data, image_folder, output_folder)
preprocess_images(val_data, image_folder, output_folder)
# Load and preprocess test data
test_data = load_annotations(test_annotation_file)
preprocess_images(test_data, image_folder, output_folder)
# Save the train, validation, and test data with attributes
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
print("Preprocessing complete.")
    # The mean shape is calculated to provide a standard reference for aligning all faces in the dataset,
    # ensuring consistent positioning of facial landmarks across processed images.
    landmarks = data.iloc[:, :196].to_numpy()
    mean_shape = landmarks.reshape(-1, 98, 2).mean(axis=0)
    return mean_shape
    # Utilizes a partial affine transformation to align facial landmarks of the image to a pre-defined mean shape.
    # This method is critical for normalizing facial images in studies that require consistent orientation and scaling.
    src_landmarks = src_landmarks.astype(np.float32)
    dst_landmarks = dst_landmarks.astype(np.float32)
    tform, inliers = cv2.estimateAffinePartial2D(src_landmarks, dst_landmarks)
    if tform is not None:
        transformed_image = cv2.warpAffine(image, tform, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        transformed_landmarks = cv2.transform(np.array([src_landmarks]), tform)[0]
        x, y, w, h = cv2.boundingRect(transformed_landmarks.astype(np.int32))
        # Ensures that the cropping does not extend beyond the image boundary, a common issue when transforming images.
        x, y, w, h = max(x, 0), max(y, 0), min(w, image.shape[1]), min(h, image.shape[0])
        cropped_image = transformed_image[y:y + h, x:x + w]
        return cropped_image
    else:
        return None
    # Processes each image in the dataset for alignment, crucial for maintaining dataset integrity and ensuring
    # that subsequent machine learning models train on well-prepared data.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, row in data.iterrows():
        image_path = os.path.join(image_folder, row['image_name'])
        image = cv2.imread(image_path)
        if image is not None:
            landmarks = row[:196].values.reshape(98, 2)
            aligned_image = align_face(image, landmarks, mean_shape)
            if aligned_image is not None and aligned_image.size > 0:
                output_path = os.path.join(output_folder, row['image_name'].split('/')[-1])
                aligned_image_RGB = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
                Image.fromarray(aligned_image_RGB.astype(np.uint8)).save(output_path)
            else:
                print("Failed to align image or resulting image is empty.")
        else:
            print(f"Image not found: {image_path}")
# Setup for aligning and saving transformed datasets for training, validation, and testing
mean_shape = calculate_mean_shape(train_data)  
align_dataset(train_data, image_folder, transformed_folder, mean_shape)
align_dataset(val_data, image_folder, transformed_folder, mean_shape)
align_dataset(test_data, image_folder, transformed_folder, mean_shape)
    """
    Retrieve all images and labels from a generator object. This function iterates over the generator,
    collecting images and labels in two separate lists, then stacking them into numpy arrays.
    Useful for aggregating data from batch-wise processing to a single dataset.
    """
    images, labels = [], []
    for i in range(len(generator)):
        img, lbl = generator.next()
        images.append(img)
        labels.append(lbl)
    return np.vstack(images), np.vstack(labels)
    """
    Load annotations from a CSV file where each line corresponds to an image and its attributes.
    The function assumes a specific format: landmarks, bounding box coordinates, and class labels
    are in the initial columns, followed by the image filename. The function restructures the DataFrame
    to separate image names and their respective attributes (like pose, expression) for easier access.
    """
    data = pd.read_csv(file_path, sep='\s+', header=None)
    num_coords = 196  # 98 landmarks x 2 for x and y coordinates
    num_rect_coords = 4
    num_attrs = 6 
    data['image_name'] = data.iloc[:, num_coords + num_rect_coords + num_attrs]
    data['image_name'] = data['image_name'].apply(lambda x: x.split('/')[-1])
    attributes_columns = num_coords + num_rect_coords + np.arange(num_attrs)
    attribute_names = ['pose', 'expression', 'illumination', 'make-up', 'occlusion', 'blur']
    for i, attr in enumerate(attribute_names):
        data[attr] = data.iloc[:, attributes_columns[i]]
    return data[['image_name'] + attribute_names]
    """
    Prepare image data generators for training and validation. This function handles data augmentation,
    normalization, and the creation of training, validation, and potentially augmented data generators.
    If additional augmented data is provided, it merges it with the original training data to enhance model training.
    """
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_dataframe(
        dataframe=data[data['subset'] == 'train'],
        directory=image_folder,
        x_col='image_name',
        y_col=['pose', 'expression', 'illumination', 'make-up', 'occlusion', 'blur'],
        target_size=target_size,
        color_mode='rgb',
        class_mode='raw',
        batch_size=batch_size,
        shuffle=True
    )
    val_gen = datagen.flow_from_dataframe(
        dataframe=data[data['subset'] == 'val'],
        directory=image_folder,
        x_col='image_name',
        y_col=['pose', 'expression', 'illumination', 'make-up', 'occlusion', 'blur'],
        target_size=target_size,
        color_mode='rgb',
        class_mode='raw',
        batch_size=batch_size,
        shuffle=True
    )
    test_gen = ''
    '''datagen.flow_from_dataframe(
    dataframe=data[data['subset'] == 'test'],
    directory=image_folder,
    x_col='image_name',
    y_col=['pose', 'expression', 'illumination', 'make-up', 'occlusion', 'blur'],
    target_size=target_size,
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    shuffle=False 
    )'''
    if augmented_annotations is not None and augmented_folder is not None:
        aug_gen = datagen.flow_from_dataframe(
            dataframe=augmented_annotations,
            directory=augmented_folder,
            x_col='image_name',
            y_col=['pose', 'expression', 'illumination', 'make-up', 'occlusion', 'blur'],
            target_size=target_size,
            color_mode='rgb',
            class_mode='raw',
            batch_size=batch_size,
            shuffle=True
        )
        train_images, train_labels = get_all_images_from_generator(train_gen)
        aug_images, aug_labels = get_all_images_from_generator(aug_gen)
        combined_images = np.vstack((train_images, aug_images))
        combined_labels = np.vstack((train_labels, aug_labels))
        indices = np.arange(combined_images.shape[0])
        np.random.shuffle(indices)
        combined_images = combined_images[indices]
        combined_labels = combined_labels[indices]
        combined_datagen = ImageDataGenerator(rescale=1./255)
        combined_train_gen = combined_datagen.flow(combined_images, combined_labels, batch_size=batch_size)
        print(f'Number of samples in combined_train_gen: {len(combined_train_gen)}')
        print(f'Number of samples in val_gen: {len(val_gen)}')
        return train_gen, combined_train_gen, val_gen, test_gen
    return train_gen, train_gen, val_gen, test_gen
    """
    Augment minority classes in the dataset to prevent class imbalance. This function uses data augmentation
    techniques such as zooming and horizontal flipping on images belonging to minority classes.
    It saves the augmented images and their corresponding annotations in a specified folder.
    """
    if not os.path.exists(augmented_folder):
        os.makedirs(augmented_folder)
    datagen = ImageDataGenerator(
        zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    augmented_annotations = []
    minority_data = data[(data['make-up'] == 1) | (data['expression'] == 1)]
    for i, row in minority_data.iterrows():
        img_path = os.path.join(image_folder, row['image_name'])
        try:
            img = load_img(img_path, target_size=target_size)
        except:
            continue
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        aug_iter = datagen.flow(x, batch_size=1)
        for j in range(augmentation_factor):
            aug_img = next(aug_iter)[0].astype('uint8')
            aug_img_name = f'aug_{i}_{j}.png'
            aug_img_path = os.path.join(augmented_folder, aug_img_name)
            tf.keras.preprocessing.image.save_img(aug_img_path, aug_img)
            augmented_annotations.append([aug_img_name] + list(row[1:].values))
    augmented_annotations = pd.DataFrame(augmented_annotations, columns=['image_name'] + list(data.columns[1:]))
    return augmented_annotations
# Load and prepare data
data = load_annotations(annotation_file)
test_data = load_annotations(test_annotation_file)
# Split the data into training and validation subsets before augmentation
data['subset'] = np.random.choice(['train', 'val'], size=len(data), p=[0.8, 0.2])
## Uncomment to load test  data. 
#test_data['subset'] = 'test'  # Add subset label
#data = pd.concat([data, test_data], ignore_index=True)
# Augment minority classes and save to disk (run only once)
if not os.path.exists('augmented_annotations.csv'):
    augmented_annotations = augment_minority_classes(data[data['subset'] == 'train'], transformed_folder, augmented_folder)
    augmented_annotations.to_csv('augmented_annotations.csv', index=False)
else:
    augmented_annotations = pd.read_csv('augmented_annotations.csv')
# Loads and prepares data
data = load_annotations(annotation_file)
test_data = load_annotations(test_annotation_file)
# Split the data into training and validation subsets before augmentation
data['subset'] = np.random.choice(['train', 'val'], size=len(data), p=[0.8, 0.2])
#test_data['subset'] = 'test'  # Add subset label
#data = pd.concat([data, test_data], ignore_index=True)
# Augment minority classes and save to disk (run only once)
if not os.path.exists('augmented_annotations.csv'):
    augmented_annotations = augment_minority_classes(data[data['subset'] == 'train'], transformed_folder, augmented_folder)
    augmented_annotations.to_csv('augmented_annotations.csv', index=False)
else:
    augmented_annotations = pd.read_csv('augmented_annotations.csv')
# Prepares data generators
train_gen, combined_train_gen, val_gen, test_gen = prepare_data(data, transformed_folder, augmented_annotations, augmented_folder)
# This line searches through the given 'image_folder' directory and all its subdirectories to find all JPEG files.
# It builds a complete list of file paths for these images which will be used for random sampling.
image_names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_folder) for f in filenames if f.endswith('.jpg')]
selected_image_names = random.sample(image_names, 10)  
# Sets up a plotting grid with 10 rows and 3 columns. Each row will display three versions of one of the selected images.
# This visualization helps in comparing the original, bounding-box annotated, and aligned versions of each image. Making sure it's working properly.
fig, axs = plt.subplots(10, 3, figsize=(10, 25))  # Setting up the plot with three columns per row, 10 rows
for idx, image_name in enumerate(selected_image_names):
    # Extracts the filename from the full path to handle file location in different folders correctly.
    relative_path = (image_name.split('/'))[-1]
    output_image_path = os.path.join(output_folder, relative_path)
    transformed_image_path = os.path.join(transformed_folder, relative_path)
    try:
        # Displays the original image in the first column of the current row.
        axs[idx, 0].imshow(Image.open(image_name))
        axs[idx, 0].set_title('Original')
        axs[idx, 0].axis('off')
        # Displays the image with the bounding box (processed state)
        axs[idx, 1].imshow(Image.open(output_image_path))
        axs[idx, 1].set_title('With Bounding Box')
        axs[idx, 1].axis('off')
        # Displays the aligned image (final processed state) 
        axs[idx, 2].imshow(Image.open(transformed_image_path))
        axs[idx, 2].set_title('Aligned')
        axs[idx, 2].axis('off')
    except FileNotFoundError as e:
        # Handles the case where any of the expected image files cannot be found.
        print(f"Error: One of the files could not be found - {e}")
plt.tight_layout()
plt.show()
# Checking if data is imbalanced. 'expression' and 'make-up' are extremly imbalanced in training data.
positive_ratios = data[['pose', 'expression', 'illumination', 'make-up', 'occlusion', 'blur']].mean()
print(f'Positive class ratio for each output feature - training data\n{positive_ratios}\n')
# Read the image file
img = mpimg.imread('Data_correlation_matrix.png')
plt.figure(figsize=(20, 20))  # Adjust the size as needed
# Display the image
plt.imshow(img)
plt.axis('off')  # Hide the axis
plt.show()
# Load pre-trained CLIP model and processor
model_name = 'openai/clip-vit-base-patch32'
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
# Function to convert generator to dataset
    images = []
    labels = []
    for _ in range(len(generator)):
        batch = generator.next()
        for img, label in zip(batch[0], batch[1]):
            images.append(img)
            labels.append(label)
    return images, labels
# Convert generators to datasets
val_images, val_labels = generator_to_dataset(val_gen)
# Define prompts for each label
prompts = {
    "pose": ["normal pose", "large pose"],
    "expression": ["normal expression", "exaggerate expression"],
    "illumination": ["normal illumination", "extreme illumination"],
    "make-up": ["no make-up", "make-up"],
    "occlusion": ["no occlusion", "occlusion"],
    "blur": ["clear", "blur"]
}
# Function to preprocess images and make predictions
    text = prompts[label_name]
    inputs = processor(text=text, images=[Image.fromarray((img * 255).astype(np.uint8)) for img in images], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()
    return probs
# Make predictions on validation dataset
predictions = []
batch_size = 32  # Adjust batch size as needed
for i in range(0, len(val_images), batch_size):
    print (f"Processing Batch {i+1} ...")
    batch_images = val_images[i:i + batch_size]
    batch_preds = []
    for label_name in prompts.keys():
        pred = predict(batch_images, label_name)
        batch_preds.append(pred)
    batch_preds = np.array(batch_preds).transpose(1, 0, 2)
    predictions.extend(batch_preds)
# Convert predictions and labels to numpy arrays
predictions = np.array(predictions)
val_labels = np.array(val_labels)
# Compute accuracy for each label
accuracies = []
for i, label_name in enumerate(prompts.keys()):
    acc = accuracy_score(np.argmax(predictions[:, i], axis=1), val_labels[:, i])
    accuracies.append(acc)
# Print accuracies for each label
for idx, label in enumerate(prompts.keys()):
    print(f"Accuracy for {label}: {accuracies[idx]:.2f}")
# Custome objective function, to better handle this imbalanced data
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred) + tf.keras.backend.epsilon()
        focal_loss_value = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss_value)
    return focal_loss_fixed
# Function to compute metrics using numpy and sklearn
    """
    Compute accuracy and AUC metrics based on true and predicted labels. The predictions are binarized
    using the specified threshold for calculating accuracy. AUC is calculated on the raw predictions.
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        threshold (float): Threshold to binarize predictions.
    Returns:
        dict: Dictionary containing accuracy and AUC scores.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    metrics = {}
    y_pred_binary = (y_pred_flat > threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_true_flat, y_pred_binary),
        'auc': roc_auc_score(y_true_flat, y_pred_flat)
    }
    return metrics
# Plot correlation matrix for training and validation data
    """
    Plot correlation matrices for true and predicted labels to analyze relationships between different tasks.
    Args:
        y_true (dict): Dictionary of true labels for each task.
        y_pred (dict): Dictionary of predicted labels for each task.
        title (str): Title for the plots.
        thresholds (dict): Dictionary of thresholds for binarizing predictions.
        model_output_names (list): List of model output names corresponding to tasks.
    Saves:
        Correlation matrices as a PNG file and displays the plot.
    """
    concatenated_true = np.column_stack([y_true[name] for name in model_output_names])
    concatenated_pred = np.column_stack([(y_pred[name] > thresholds[name]).astype(int) for name in model_output_names])
    df_true = pd.DataFrame(concatenated_true, columns=model_output_names)
    df_pred = pd.DataFrame(concatenated_pred, columns=model_output_names)
    corr_true = df_true.corr()
    corr_pred = df_pred.corr()
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.heatmap(corr_true, annot=True, ax=axes[0], cmap='coolwarm')
    axes[0].set_title(f'{title} - True Labels Correlation')
    sns.heatmap(corr_pred, annot=True, ax=axes[1], cmap='coolwarm')
    axes[1].set_title(f'{title} - Predicted Labels Correlation')
    plt.tight_layout()
    plt.savefig(f'{title}_correlation_matrix.png')
    plt.show()
# Function to predict and collect true and predicted labels
    """
    Predicts labels for all images in the generator and collects true and predicted labels.
    Args:
        generator (tf.keras.utils.Sequence): Data generator for images.
        model (tf.keras.Model): Trained model to make predictions.
    Returns:
        tuple: True labels, predicted labels, and images as numpy arrays.
    """
    all_y_true = {name: [] for name in model.output_names}
    all_y_pred = {name: [] for name in model.output_names}
    images = []
    for _ in range(len(generator)):
        x, y_true = next(generator)
        y_pred = model.predict(x, verbose=0)
        for i, name in enumerate(model.output_names):
            all_y_true[name].append(y_true[:, i])
            all_y_pred[name].append(y_pred[i])
        images.append(x)
    # Convert lists to numpy arrays
    for name in model.output_names:
        all_y_true[name] = np.concatenate(all_y_true[name], axis=0)
        all_y_pred[name] = np.concatenate(all_y_pred[name], axis=0)
    images = np.concatenate(images, axis=0)
    return all_y_true, all_y_pred, images
        """
        Initialize ModelEvaluator with the provided model path, train generator, and validation generator.
        Loads the model, predicts on the datasets, computes metrics, and initializes thresholds.
        Args:
            model_path (str): Path to the trained model.
            train_gen (tf.keras.utils.Sequence): Training data generator.
            val_gen (tf.keras.utils.Sequence): Validation data generator.
        """
        self.model_path = model_path
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.custom_objects = {'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)}
        self.model = load_model(self.model_path, custom_objects=self.custom_objects)
        self.train_y_true, self.train_y_pred, self.train_images = predict_and_collect(self.train_gen, self.model)
        self.val_y_true, self.val_y_pred, self.val_images = predict_and_collect(self.val_gen, self.model)
        self.thresholds = {name: np.mean(self.train_y_pred[name][self.train_y_true[name] == 1]) for name in self.model.output_names}
        self.train_metrics = self.compute_metrics(self.train_y_true, self.train_y_pred)
        self.val_metrics = self.compute_metrics(self.val_y_true, self.val_y_pred)
        self.metrics_df = self.create_metrics_df()
        self.save_metrics()
        self.print_metrics()
        self.identify_performance()
        self.plot_correlation_matrices()
        """
        Compute metrics for each task based on true and predicted labels.
        Args:
            y_true (dict): Dictionary of true labels for each task.
            y_pred (dict): Dictionary of predicted labels for each task.
        Returns:
            dict: Dictionary of computed metrics for each task.
        """
        return {name: compute_metrics_np(y_true[name], y_pred[name], self.thresholds[name]) for name in self.model.output_names}
        """
        Create a DataFrame to store metrics for each task and dataset (train and validation).
        Returns:
            pd.DataFrame: DataFrame containing metrics for each task and dataset.
        """
        metrics_data = []
        for name in self.model.output_names:
            for metric in self.train_metrics[name].keys():
                metrics_data.append({
                    'Task': name,
                    'Metric': metric,
                    'Dataset': 'Train',
                    'Value': self.train_metrics[name][metric]
                })
                metrics_data.append({
                    'Task': name,
                    'Metric': metric,
                    'Dataset': 'Validation',
                    'Value': self.val_metrics[name][metric]
                })
        df = pd.DataFrame(metrics_data)
        # Pivot the DataFrame to have the desired format
        metrics_df = df.pivot_table(index='Dataset', columns=['Task', 'Metric'], values='Value')
        return metrics_df
        """
        Save the metrics DataFrame to a CSV file for further analysis.
        """
        self.metrics_df.to_csv('model_metrics.csv', index=False)
        """
        Print the metrics DataFrame to the console for review.
        """
        print(self.metrics_df)
        """
        Identify the best and worst performing images for training and validation datasets.
        """
        self.train_good_indices, self.train_bad_indices = {}, {}
        self.val_good_indices, self.val_bad_indices = {}, {}
        for name in self.model.output_names:
            self.train_good_indices[name], self.train_bad_indices[name] = identify_performance(self.train_y_true[name], self.train_y_pred[name], self.thresholds[name])
            self.val_good_indices[name], self.val_bad_indices[name] = identify_performance(self.val_y_true[name], self.val_y_pred[name], self.thresholds[name])
        """
        Plot correlation matrices for the true and predicted labels in training and validation datasets.
        """
        plot_correlation_matrix(self.train_y_true, self.train_y_pred, 'Training', self.thresholds, self.model.output_names)
        plot_correlation_matrix(self.val_y_true, self.val_y_pred, 'Validation', self.thresholds, self.model.output_names)
        """
        Display images in a table format with labels on top.
        Args:
            indices (list): List of image indices to display.
            images (np.array): Array of images.
            title (str): Title for the images.
        """
        plt.figure(figsize=(20, 10))
        for i, idx in enumerate(indices):
            plt.subplot(1, len(indices), i+1)
            plt.imshow(array_to_img(images[idx]))
            plt.title(title)
            plt.axis('off')
        plt.show()
        """
        Print and display the top 5 best and worst performing images for each task in training and validation datasets.
        """
        print("\nTop 5 Best Performing Images (Train):\n")
        for name in self.model.output_names:
            print(f"Task: {name}")
            print("Best Images:")
            self.display_images(self.train_good_indices[name], self.train_images, f"Train Best - {name}")
            print("Worst Images:")
            self.display_images(self.train_bad_indices[name], self.train_images, f"Train Worst - {name}")
        print("\nTop 5 Best Performing Images (Validation):\n")
        for name in self.model.output_names:
            print(f"Task: {name}")
            print("Best Images:")
            self.display_images(self.val_good_indices[name], self.val_images, f"Val Best - {name}")
            print("Worst Images:")
            self.display_images(self.val_bad_indices[name], self.val_images, f"Val Worst - {name}")
# Define a multi-task learning model
    """
    Constructs a multi-task model using convolutional neural networks (CNNs) to predict multiple
    attributes from facial images. The model outputs six attributes: pose, expression, illumination,
    make-up, occlusion, and blur.
    Returns:
        model (tf.keras.Model): Compiled Keras model for multi-task learning.
    """
    inputs = Input(shape=(224, 224, 3))
    # Initial convolutional layer with 32 filters
    x = Conv2D(32, (4, 4), activation='relu', padding='same')(inputs)
    x_conv = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    # Max pooling layer to downsample the feature maps
    x = MaxPooling2D(2, 2)(x_conv)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = Flatten()(x)
    # Output layers for each task, using sigmoid activation for binary classification
    pose_output = Dense(1, activation='sigmoid', name='pose')(x)
    expression_output = Dense(1, activation='sigmoid', name='expression')(x)
    illumination_output = Dense(1, activation='sigmoid', name='illumination')(x)
    make_up_output = Dense(1, activation='sigmoid', name='make_up')(x)
    occlusion_output = Dense(1, activation='sigmoid', name='occlusion')(x)
    blur_output = Dense(1, activation='sigmoid', name='blur')(x)
    # Compile the model with Adam optimizer and focal loss
    model = Model(inputs=inputs, outputs=[pose_output, expression_output, illumination_output, make_up_output, occlusion_output, blur_output])
    model.compile(optimizer=Adam(learning_rate=0.005), 
                  loss=focal_loss(gamma=2., alpha=0.25), 
                  metrics=['accuracy'])
    return model
# Define an attention block to improve feature extraction
    """
    Creates an attention block consisting of multiple convolutional layers with different
    kernel sizes to capture varied spatial information. Outputs are concatenated and passed
    through an additional convolutional layer.
    Args:
        inputs (tf.Tensor): Input tensor.
        filters (int): Number of filters for convolutional layers.
    Returns:
        tf.Tensor: Output tensor after attention block processing.
    """
    g1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
    g1 = MaxPooling2D(pool_size=(2, 2))(g1)
    g2 = Conv2D(filters, (2, 2), padding='same', activation='relu')(inputs)
    g2 = MaxPooling2D(pool_size=(2, 2))(g2)
    g3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    g3 = MaxPooling2D(pool_size=(2, 2))(g3)
    # Concatenate the outputs of the convolutional layers
    merged = Concatenate()([g1, g2, g3])
    # Additional convolutional layer for merged output
    g4 = Conv2D(filters, (1, 1), padding='same', activation='relu')(merged)
    return g4
# Function to build the attention-based model
    """
    Constructs a multi-task learning model incorporating attention mechanisms to enhance
    feature extraction. The model predicts six attributes from facial images.
    Args:
        input_shape (tuple): Shape of the input images.
    Returns:
        model (tf.keras.Model): Compiled Keras model with attention mechanisms.
    """
    inputs = Input(shape=input_shape)
    # Attention block 1
    attention1 = attention_block(inputs, 16)
    # Convolutional block 1
    conv1 = Conv2D(32, (2, 2), padding='same', activation='relu')(attention1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # Attention block 2
    attention2 = attention_block(pool1, 32)
    # Convolutional block 2
    conv2 = Conv2D(64, (2, 2), padding='same', activation='relu')(attention2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # Flatten the output
    gap = Flatten()(pool2)
    # Fully connected layers for each task with sigmoid activation
    pose_output = Dense(1, activation='sigmoid', name='pose')(gap)
    expression_output = Dense(1, activation='sigmoid', name='expression')(gap)
    illumination_output = Dense(1, activation='sigmoid', name='illumination')(gap)
    make_up_output = Dense(1, activation='sigmoid', name='make_up')(gap)
    occlusion_output = Dense(1, activation='sigmoid', name='occlusion')(gap)
    blur_output = Dense(1, activation='sigmoid', name='blur')(gap)
    model = Model(inputs=inputs, outputs=[pose_output, expression_output, illumination_output, make_up_output, occlusion_output, blur_output])
    # Compile the model with Adam optimizer and focal loss
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=focal_loss(gamma=2., alpha=0.25), 
                  metrics=['accuracy'])
    return model
# Function to get embedding from DeepFace
    """
    Extracts embeddings for a given image using DeepFace with VGG-Face model.
    Args:
        img_path (str): Path to the image file.
    Returns:
        np.array: Embedding vector for the image.
    """
    embeddings = DeepFace.represent(img_path=img_path, model_name="VGG-Face", enforce_detection=False)
    return np.array(embeddings[0]['embedding'])
# Define a model using DeepFace embeddings
    """
    Constructs a multi-task model using DeepFace embeddings as input. The model processes the embeddings
    and predicts six facial attributes.
    Args:
        input_shape (tuple): Shape of the input embeddings.
    Returns:
        model (tf.keras.Model): Compiled Keras model using DeepFace embeddings.
    """
    input_layer = Input(shape=(50176, 3))
    x = Reshape((224, 224, 3))(input_layer)
    x = Conv2D(128, (2, 2), padding='same', activation='relu')(x)
    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Conv2D(256, (2, 2), padding='same', activation='relu')(x)
    x = AveragePooling2D(pool_size=(4, 4))(x)
    gap = Flatten()(x)
    # Fully connected layers for each task with sigmoid activation
    pose_output = Dense(1, activation='sigmoid', name='pose')(gap)
    expression_output = Dense(1, activation='sigmoid', name='expression')(gap)
    illumination_output = Dense(1, activation='sigmoid', name='illumination')(gap)
    make_up_output = Dense(1, activation='sigmoid', name='make_up')(gap)
    occlusion_output = Dense(1, activation='sigmoid', name='occlusion')(gap)
    blur_output = Dense(1, activation='sigmoid', name='blur')(gap)
    model = Model(inputs=input_layer, outputs=[pose_output, expression_output, illumination_output, make_up_output, occlusion_output, blur_output])
    # Compile the model with Adam optimizer and focal loss
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=focal_loss(gamma=2., alpha=0.25), 
                  metrics=['accuracy'])
    return model
model = attention_model()
# or load model
#model = load_model('best_att_no_aug_new.h5', custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)}) 
#tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0001)
checkpoint = ModelCheckpoint('best_att_no_aug_new.h5',  monitor='val_loss', save_best_only=True, mode='min')
model.summary()
history = model.fit(train_gen, validation_data=val_gen, epochs=500, callbacks=[checkpoint])
Att_Aug_model = attention_model()
# or load model
#Att_Aug_model = load_model('best_Attmodel_with_Aug.h5', custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)}) 
#tf.keras.backend.set_value(Att_Aug_model.optimizer.learning_rate, 0.0001)
checkpoint = ModelCheckpoint('best_Attmodel_with_Aug.h5',  monitor='val_loss', save_best_only=True, mode='min')
Att_Aug_model.summary()
history = Att_Aug_model.fit(combined_train_gen, validation_data=val_gen, epochs=500, callbacks=[checkpoint])
x_sample, _ = next(train_gen)
sample_img = x_sample[0]  # Use the first image in the batch
# Save the sample image temporarily to get its path
sample_img_path = 'temp_img.jpg'
tf.keras.preprocessing.image.save_img(sample_img_path, sample_img)
# Generate the embedding for the sample image
sample_embedding = get_embedding(sample_img_path)
input_shape = (sample_embedding.shape[0],)
# Build the model with the shape of embeddings
model = deepFace_model(input_shape=input_shape)
# Print model summary
model.summary()
train_gen.batch_size = 16
val_gen.batch_size = 16
# Train the model
checkpoint = ModelCheckpoint('best_deep_face.h5', save_best_only=False)
history = model.fit(train_gen, validation_data=val_gen, epochs=500,  callbacks=[checkpoint])
model_path = 'best_att_no_aug_new.h5'
evaluator = ModelEvaluator(model_path, train_gen, val_gen)#, test_gen)
evaluator.print_top_bottom_images()
# Attention Model
model_path = 'best_Attmodel_with_Aug.h5'
train_gen.batch_size , val_gen.batch_size = 64 , 64
evaluator = ModelEvaluator(model_path, train_gen, val_gen)
evaluator.print_top_bottom_images()
