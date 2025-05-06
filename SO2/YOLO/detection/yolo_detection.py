if __name__ == "__main__":
    import torch
    from ultralytics import YOLO

    # Load YOLO model
    model = YOLO("yolo11l.pt")

    # Train model with automatic dataset split
    model.train(
        data=r"C:\Users\amirs\Desktop\MAS500\MAS500\detection\detection.yaml",  # Use the YAML file
        epochs=500,
        batch=64,
        imgsz=256,
        device="cuda",
        workers=8,
        patience=50,
        split=0.8  # Automatically split: 80% train, 20% validation
    )
