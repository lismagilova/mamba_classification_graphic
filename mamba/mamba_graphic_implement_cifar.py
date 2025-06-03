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

        # Параметры SSM (A, B, C, D)
        self.A = nn.Parameter(torch.randn(self.d_state) * 0.1 - 1.0)
        self.B_proj = nn.Linear(d_model, self.d_state)
        self.C_proj = nn.Linear(self.d_state, d_model)
        self.D = nn.Parameter(torch.ones(d_model))
        self.time_proj = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        delta = torch.sigmoid(self.time_proj(x)) * 0.99 + 0.01  # [batch, seq_len, 1]

        B = self.B_proj(x)  # [batch, seq_len, d_state]

        # A: [d_state] -> A_exp: [batch, seq_len, d_state]
        A_exp = torch.exp(self.A.unsqueeze(0).unsqueeze(0) * delta)

        h = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            h = A_exp[:, t] * h + B[:, t]  # h_t = A_t * h_{t-1} + B_t (x_t уже в B_t)
            out_t = self.C_proj(h)
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, d_model]
        y = outputs + x * self.D.view(1, 1, -1)
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
        mlp_in_val = self.mlp_in(self.norm2(x))
        mlp_hidden, gate = mlp_in_val.chunk(2, dim=-1)
        mlp_hidden = mlp_hidden * F.silu(gate)
        x = x + self.dropout(self.mlp_out(mlp_hidden))
        return x


class MambaImageClassifier(nn.Module):
    def __init__(self,
                 img_size=32,
                 in_channels=3,
                 patch_size=4,
                 d_model=128,
                 d_state=16,
                 depth=4,
                 num_classes=10,
                 dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_proj = nn.Sequential(
            Reshape((-1, in_channels, img_size, img_size)),
            nn.Unfold(kernel_size=patch_size, stride=patch_size),
            PermuteAndReshape((-1, self.patch_dim, self.num_patches), (0, 2, 1)),  # (B, C*P*P, N) -> (B, N, C*P*P)
            nn.Linear(self.patch_dim, d_model)
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embedding, std=0.02)
        if isinstance(self.classifier, nn.Linear):
            nn.init.xavier_uniform_(self.classifier.weight)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)

    def forward(self, imgs):
        x = self.patch_proj(imgs)
        x = x + self.pos_embedding
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global Average Pooling
        logits = self.classifier(x)
        return logits


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        if self.shape[0] == -1:
            return x.reshape(x.shape[0], *self.shape[1:])
        return x.reshape(self.shape)


class PermuteAndReshape(nn.Module):
    def __init__(self, target_shape_before_permute, permute_dims):
        super().__init__()
        self.target_shape_before_permute = target_shape_before_permute
        self.permute_dims = permute_dims

    def forward(self, x):
        if self.target_shape_before_permute[0] == -1:
            x = x.reshape(x.shape[0], *self.target_shape_before_permute[1:])
        else:
            x = x.reshape(self.target_shape_before_permute)
        return x.permute(*self.permute_dims)


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def run_demo():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MambaImageClassifier(
        img_size=32, in_channels=3, patch_size=4,
        d_model=64, d_state=16, depth=2, num_classes=10, dropout=0.1
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print("Демонстрационное обучение на одном батче CIFAR-10...")
    model.train()

    num_demo_batches = 5
    data_iter = iter(train_loader)
    for i in range(num_demo_batches):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Демо-батч {i + 1}/{num_demo_batches}, Loss: {loss.item():.4f}")

    visualize_predictions(model, test_loader, num_samples=5, device=device)
    return model


def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_predictions(model, test_loader, num_samples=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()

    dataiter = iter(test_loader)
    try:
        images, labels = next(dataiter)
    except StopIteration:
        print("Не удалось загрузить данные из test_loader для визуализации.")
        return

    images_subset = images[:num_samples].to(device)
    labels_subset = labels[:num_samples]

    with torch.no_grad():
        outputs = model(images_subset)
        _, predicted = torch.max(outputs, 1)

    fig = plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i + 1, xticks=[], yticks=[])
        img = images_subset[i].cpu()
        img = unnormalize(img, CIFAR_MEAN, CIFAR_STD)
        img = img.permute(1, 2, 0)  # C, H, W -> H, W, C
        img = torch.clamp(img, 0, 1)

        ax.imshow(img.numpy())
        true_label = cifar10_classes[labels_subset[i]]
        pred_label = cifar10_classes[predicted[i].item()]
        color = "green" if predicted[i] == labels_subset[i] else "red"
        ax.set_title(f"Ист: {true_label}\nПред: {pred_label}", color=color, fontsize=10)

    plt.tight_layout()
    plt.show()

    print("\nПрогноз модели на нескольких примерах:")
    correct = 0
    for i in range(num_samples):
        result = "✓" if predicted[i] == labels_subset[i] else "✗"
        correct += (predicted[i] == labels_subset[i]).item()
        print(f"Изображение {i + 1}: Истинное = {cifar10_classes[labels_subset[i]]}, "
              f"Прогноз = {cifar10_classes[predicted[i].item()]}, {result}")
    print(f"\nТочность на {num_samples} примерах: {correct / num_samples:.2f}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    batch_size = 64 if device == torch.device('cuda') else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=True if device == torch.device('cuda') else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2,
                            pin_memory=True if device == torch.device('cuda') else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2,
                             pin_memory=True if device == torch.device('cuda') else False)

    model = MambaImageClassifier(
        img_size=32, in_channels=3, patch_size=4,
        d_model=128,
        d_state=16,
        depth=4,
        num_classes=10,
        dropout=0.1
    )

    epochs = 50

    print(f"Планируется {epochs} эпох обучения.")

    trained_model = train_full_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device
    )

    accuracy = evaluate_model(trained_model, test_loader, device)
    visualize_predictions(trained_model, test_loader, num_samples=10, device=device)
    return trained_model, accuracy


def train_full_model(model, train_loader, val_loader, epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model = model.to(device)
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Полное обучение на {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total_samples = 0, 0, 0

        pbar_train = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{epochs} [Обучение]", leave=False)
        for images, labels in pbar_train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            pbar_train.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        train_loss /= total_samples
        train_acc = train_correct / total_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss, val_correct, total_val_samples = 0, 0, 0
        pbar_val = tqdm(val_loader, desc=f"Эпоха {epoch + 1}/{epochs} [Валидация]", leave=False)
        with torch.no_grad():
            for images, labels in pbar_val:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)
                pbar_val.set_postfix({'loss': loss.item()})

        val_loss /= total_val_samples
        val_acc = val_correct / total_val_samples
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        print(f"Эпоха {epoch + 1}/{epochs} - "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mamba_cifar10_model.pth')
            print(f"  Сохранена лучшая модель с Val Acc: {best_val_acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Потери обучения')
    plt.plot(val_losses, label='Потери валидации')
    plt.xlabel('Эпоха');
    plt.ylabel('Потери');
    plt.legend()
    plt.title('График потерь')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Точность обучения')
    plt.plot(val_accs, label='Точность валидации')
    plt.xlabel('Эпоха');
    plt.ylabel('Точность');
    plt.legend()
    plt.title('График точности')

    plt.tight_layout()
    plt.show()

    model.load_state_dict(torch.load('best_mamba_cifar10_model.pth'))
    return model


def evaluate_model(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    correct, total = 0, 0

    pbar_test = tqdm(test_loader, desc="Тестирование", leave=False)
    with torch.no_grad():
        for images, labels in pbar_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Точность на тестовом наборе CIFAR-10: {accuracy:.4f} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    # демонстрация
    # demo_model = run_demo()

    # полное обучение
    trained_model, final_accuracy = main()
    print(f"--- Финальная точность на тесте: {final_accuracy:.4f} ---")

    # код для сохранения/загрузки модели
    torch.save(trained_model.state_dict(), 'mamba_cifar10_final.pth')
    model_loaded = MambaImageClassifier(...)
    model_loaded.load_state_dict(torch.load('mamba_cifar10_final.pth'))
    model_loaded.to(device)
    evaluate_model(model_loaded, test_loader, device)
