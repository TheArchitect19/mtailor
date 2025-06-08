import torch
from torchvision import models, transforms
from PIL import Image
import io, base64

# Step 1: Define same architecture as used in training
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Step 2: Load the saved weights
state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Step 3: Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])
])

# Step 4: Define handler function
def run(event, context=None):
    """
    Input: { "image_base64": "<...>" }
    Output: { "label": int, "score": float }
    """
    image_data = base64.b64decode(event["image_base64"])
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        label = torch.argmax(probs).item()
        score = probs[label].item()

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return {"label": classes[label], "score": score}
