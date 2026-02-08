import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "saved_model\image_adversarial_model.pt"
IMAGE_PATH = "data/test_images/0.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model_state"])
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def test_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1)[0]

    clean_prob = probs[0].item()
    adv_prob = probs[1].item()

    if adv_prob > 0.7:
        action = "BLOCK"
        label = "ADVERSARIAL"
        conf = adv_prob
    elif adv_prob > 0.4:
        action = "WARNING"
        label = "SUSPICIOUS"
        conf = adv_prob
    else:
        action = "ALLOW"
        label = "CLEAN"
        conf = clean_prob

    print("Prediction:", label)
    print("Confidence:", f"{conf*100:.2f}%")
    print("Action:", action)

test_image(IMAGE_PATH)
