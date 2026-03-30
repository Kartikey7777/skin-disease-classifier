import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

# ─── Config ───────────────────────────────────────────
MODEL_PATH  = "model/best_model.pth"
NUM_CLASSES = 7
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesion"
]

# ─── Transform ────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Load Model ───────────────────────────────────────
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

# ─── Predict ──────────────────────────────────────────
def predict(image_path):
    model = load_model()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = probs.argmax().item()

    print(f"\n🔍 Prediction: {CLASS_NAMES[pred]}")
    print(f"📊 Confidence: {probs[pred]*100:.2f}%")
    print("\nAll class probabilities:")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        print(f"  {name}: {prob*100:.2f}%")

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    predict(img_path)