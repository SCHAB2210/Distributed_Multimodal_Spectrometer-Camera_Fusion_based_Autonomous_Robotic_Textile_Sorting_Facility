if __name__ == "__main__":
    import torch
    from ultralytics import YOLO

    # Load model
    model = YOLO("yolo11m-cls.pt")

   
    model.train(
        data=r"C:\Users\amirs\Desktop\backup\camera\models\dataset\material\material_HR",
        epochs=200,  # Reduced for quick testing
        batch=128,
        imgsz=256,
        device="cuda",
        workers=8,  # Prevent multiprocessing issues
        patience=20  
    )
