import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406],
                             std=[.229, .224, .225]),
    ])
    train_ds = datasets.CIFAR10(root="data/",
                                train=True,
                                download=True,
                                transform=transform)
    train_loader = DataLoader(train_ds,
                              batch_size=32,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # we can also use SGD as optimiser for less complex

    
    epochs = 1
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for idx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % 100 == 0:
                avg = running_loss / 100
                print(f"[Epoch {epoch+1}/{epochs}] Step {idx:4d}  Loss: {avg:.4f}")
                running_loss = 0.0


    torch.save(model.state_dict(), "model.pth")
    print("✅ Saved trained model to model.pth")

if __name__ == "__main__":
    main()
