from ultralytics import YOLO
import os

# Ensure the weights directory exists
os.makedirs("weights", exist_ok=True)

# Load a model
print("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")  # load a pretrained model

# Train the model
print("Starting training...")
results = model.train(
    data="data.yaml",
    epochs=10,
    imgsz=416,
    batch=8,
    save=True,
    verbose=True,
    device="cpu"
)

# Get the best model path from training results
try:
    best_model_path = results.best if hasattr(results, 'best') else None
    print(f"Best model path from results: {best_model_path}")
except:
    pass

# Manually save the trained model
print("Saving model...")
model.save("weights/best.pt")
print("Model saved to weights/best.pt")

# Verify the file exists
if os.path.exists("weights/best.pt"):
    print(f"Model file size: {os.path.getsize('weights/best.pt') / 1024 / 1024:.2f} MB")
else:
    print("ERROR: Model file was not saved!")

print("Training completed!")
