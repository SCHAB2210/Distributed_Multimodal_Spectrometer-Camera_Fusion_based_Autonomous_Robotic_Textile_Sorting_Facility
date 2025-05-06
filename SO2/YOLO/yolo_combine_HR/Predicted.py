from ultralytics import YOLO
import cv2
import torch

def classify_image(model_path, image_path):
    # Load YOLOv8 model
    model = YOLO(model_path)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Retrieve class names from the model
    class_labels = model.names  # Model should contain class names as a dictionary
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image.")
        return
    
    # Run inference
    results = model(image)
    
    # Get classification result
    for result in results:
        probs = result.probs  # Get probabilities for each class
        sorted_indices = probs.data.argsort(descending=True)  # Sort indices by confidence
        
        print("Class Predictions:")
        for idx in sorted_indices:
            idx = idx.item()  # Convert tensor to integer
            class_label = class_labels[idx] if idx in class_labels else f"Unknown ({idx})"
            confidence = probs.data[idx].item()
            print(f"Class: {class_label}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    # Manually specify paths
    model_path = r"C:\Users\amirs\Desktop\MAS500\model 1\runs\classify\train\weights\best.pt"  # Replace with actual path
    image_path = r"C:\Users\amirs\Desktop\MAS500\model 1\imagetest\p_test.jpg"  # Replace with actual path
    
    
    classify_image(model_path, image_path)
