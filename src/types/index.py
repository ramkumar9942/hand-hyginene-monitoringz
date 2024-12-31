from typing import List, Tuple, Any

# Define a type for image data
ImageData = Any  # Placeholder for image data type

# Define a type for labels
Label = int  # Assuming labels are integers

# Define a type for a dataset
Dataset = List[Tuple[ImageData, Label]]  # List of tuples containing image data and corresponding labels

# Define a type for model predictions
Prediction = Tuple[Label, float]  # Tuple containing predicted label and confidence score

# Exporting types for use in other modules
__all__ = ['ImageData', 'Label', 'Dataset', 'Prediction']