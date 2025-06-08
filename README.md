# mtailor



Here is the updated `README.md` **without emojis**:

---

```markdown
# mtailor — ResNet18 CIFAR-10 Classifier

This project demonstrates how to train a ResNet18 model on the CIFAR-10 dataset using PyTorch and then serve it as an inference API that accepts base64-encoded images.

---

## Project Structure

```

mtailor/
│
├── model.pth                  # Trained model weights
├── train.py                   # Script to train ResNet18 on CIFAR-10
├── inference.py               # Inference handler (used in API deployment)
├── cerebrium.toml             # Cerebrium deployment config
├── requirements.txt           # Python dependencies
└── README.md                  # This file

````

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ayushkumar/mtailor.git
cd mtailor
````

---

### 2. Create & Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

If you haven't already generated `requirements.txt`, you can do:

```bash
pip install torch torchvision pillow
pip freeze > requirements.txt
```

Otherwise, install from it:

```bash
pip install -r requirements.txt
```

---

## Training the Model

To train the ResNet18 model on CIFAR-10 and save weights to `model.pth`:

```bash
python train.py
```

This will:

* Download CIFAR-10
* Train for 1 epoch
* Save the model to `model.pth`

---

## Running Inference Locally

To test the model locally using a base64 image input:

```python
# In Python shell or script

from inference import run
with open("sample.jpg", "rb") as f:
    import base64
    img_base64 = base64.b64encode(f.read()).decode("utf-8")

result = run({"image_base64": img_base64})
print(result)
```

---

## Deploying as an API (Cerebrium.ai)

Make sure you have the Cerebrium CLI installed.

### 1. Authenticate:

```bash
cerebrium login
```

### 2. Deploy the app:

```bash
cerebrium deploy
```

This will:

* Package `inference.py` as the handler
* Upload your `model.pth`
* Serve your model as a REST API

### 3. Test your endpoint

```bash
curl -X POST https://api.cortex.cerebrium.ai/v4/<project_id>/<app_name>/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_token>" \
  -d '{"image_base64": "<base64_image_data>"}'
```

---

## Classes

Your model classifies images into:

```
'plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse',
'ship', 'truck'
```

---

## Author

Built by Ayush Kumar

---

# My API

```
https://api.cortex.cerebrium.ai/v4/p-94f8d376/mtailor/run
{
  "event": {
    "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
  }
}

```
