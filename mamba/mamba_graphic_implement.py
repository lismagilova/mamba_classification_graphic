import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


class SelectiveScanModule(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A = nn.Parameter(torch.randn(self.d_state) * 0.1 - 1.0)
        self.B_proj = nn.Linear(d_model, self.d_state)
        self.C_proj = nn.Linear(self.d_state, d_model)
        self.D = nn.Parameter(torch.ones(d_model))
        self.time_proj = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        delta = torch.sigmoid(self.time_proj(x)) * 0.99 + 0.01
        B = self.B_proj(x)
        A_discrete = torch.exp(self.A.unsqueeze(0).unsqueeze(0) * delta)
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = A_discrete[:, t] * h + B[:, t]
            out = self.C_proj(h)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        y = x * self.D.view(1, 1, -1) + outputs
        y = self.dropout(self.out_proj(y))
        return y


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ssm = SelectiveScanModule(d_model, d_state=d_state, dropout=dropout)
        self.mlp_in = nn.Linear(d_model, 2 * self.d_hidden)
        self.mlp_out = nn.Linear(self.d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        mlp_in = self.mlp_in(self.norm2(x))
        mlp_hidden, gate = mlp_in.chunk(2, dim=-1)
        mlp_hidden = mlp_hidden * F.silu(gate)
        x = x + self.dropout(self.mlp_out(mlp_hidden))
        return x


class MambaImageClassifier(nn.Module):
    def __init__(self, img_size=28, in_channels=1, patch_size=4, d_model=64, d_state=16, depth=2, num_classes=10):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_proj = nn.Sequential(
            Reshape((-1, in_channels, img_size, img_size)),
            nn.Unfold(kernel_size=patch_size, stride=patch_size),
            Reshape((-1, self.num_patches, self.patch_dim)),
            nn.Linear(self.patch_dim, d_model)
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state=d_state) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, imgs):
        x = self.patch_proj(imgs)
        x = x + self.pos_embedding
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape) if self.shape[0] == -1 else x.reshape((x.shape[0],) + self.shape)


def run_demo():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    model = MambaImageClassifier(
        img_size=28,
        in_channels=1,
        patch_size=4,
        d_model=32,
        d_state=8,
        depth=2,
        num_classes=10
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    print("Демонстрационное обучение на одном батче...")
    model.train()
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Итерация {i + 1}/5, Loss: {loss.item():.4f}")
    visualize_predictions(model, test_loader, num_samples=5, device=device)
    return model


def visualize_predictions(model, test_loader, num_samples=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    fig = plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i + 1)
        img = images[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray')
        color = "green" if predicted[i] == labels[i] else "red"
        ax.set_title(f"Истинное: {labels[i]}\nПрогноз: {predicted[i].item()}", color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    print("\nПрогноз модели:")
    correct = 0
    for i in range(num_samples):
        result = "✓" if predicted[i] == labels[i] else "✗"
        correct += (predicted[i] == labels[i]).item()
        print(f"Изображение {i + 1}: Истинное значение = {labels[i]}, Прогноз = {predicted[i]}, {result}")
    print(f"\nТочность на {num_samples} примерах: {correct / num_samples:.2f}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    batch_size = 32 if device == 'cpu' else 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = MambaImageClassifier(
        img_size=28,
        in_channels=1,
        patch_size=4,
        d_model=64,
        d_state=16,
        depth=3,
        num_classes=10
    )
    train_full_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2 if device == 'cpu' else 5,
        device=device
    )
    accuracy = evaluate_model(model, test_loader, device)
    visualize_predictions(model, test_loader, num_samples=10, device=device)
    return model, accuracy


def train_full_model(model, train_loader, val_loader, epochs=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model = model.to(device)
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    print(f"Полное обучение на {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        total_samples = 0
        for images, labels in tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{epochs} [Обучение]"):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            total_samples += batch_size
        train_loss /= total_samples
        train_acc = train_correct / total_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        model.eval()
        val_loss, val_correct = 0, 0
        total_val_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Эпоха {epoch + 1}/{epochs} [Валидация]"):
                images, labels = images.to(device), labels.to(device)
                batch_size = images.size(0)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val_samples += batch_size
        val_loss /= total_val_samples
        val_acc = val_correct / total_val_samples
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mamba_model.pth')
        print(
            f"Эпоха {epoch + 1}/{epochs} - Потери обучения: {train_loss:.4f}, Точность обучения: {train_acc:.4f}, Потери валидации: {val_loss:.4f}, Точность валидации: {val_acc:.4f}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Потери обучения')
    plt.plot(val_losses, label='Потери валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Точность обучения')
    plt.plot(val_accs, label='Точность валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return model


def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Тестирование"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Точность на тестовом наборе: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    main()
