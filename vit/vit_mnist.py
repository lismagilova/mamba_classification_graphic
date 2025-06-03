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
import time


class Reshape(nn.Module):
    def __init__(self, shape): super().__init__(); self.shape = shape

    def forward(self, x):
        if self.shape[0] == -1: return x.reshape(x.shape[0], *self.shape[1:])
        return x.reshape(self.shape)


class ViTBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_hidden_dim, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10, d_model=64, depth=3, num_heads=4,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.depth = depth
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_proj = nn.Sequential(Reshape((-1, in_channels, img_size, img_size)),
                                        nn.Unfold(kernel_size=patch_size, stride=patch_size),
                                        Reshape((-1, num_patches, patch_dim)), nn.Linear(patch_dim, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.dropout_embed = nn.Dropout(dropout)
        self.transformer_encoder = nn.ModuleList(
            [ViTBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        for module_proj in self.patch_proj.modules():
            if isinstance(module_proj, nn.Linear):
                nn.init.xavier_uniform_(module_proj.weight)
                if module_proj.bias is not None: nn.init.zeros_(module_proj.bias)

    def forward(self, img):
        B = img.shape[0]
        x = self.patch_proj(img)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout_embed(x)
        for blk in self.transformer_encoder: x = blk(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device); initial_memory = torch.cuda.memory_allocated(
        device)
    for images, labels in tqdm(loader, desc="Обучение", leave=False, ncols=100):
        images, labels = images.to(device), labels.to(device)
        batch_size_current = images.size(0)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * batch_size_current
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    if scheduler: scheduler.step()
    avg_loss = epoch_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    peak_memory_epoch = 0
    if device.type == 'cuda': peak_memory_epoch = torch.cuda.max_memory_allocated(device) - initial_memory
    return avg_loss, accuracy, peak_memory_epoch


def validate_model_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Валидация/Тест", leave=False, ncols=100):
            images, labels = images.to(device), labels.to(device)
            batch_size_current = images.size(0)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * batch_size_current
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = epoch_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def benchmark_inference_speed(model, loader, device, num_warmup_batches=5, num_benchmark_batches=20):
    model.eval()
    model = model.to(device)

    times = []
    total_samples_processed = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_warmup_batches: break
            _ = model(images.to(device))
            if device.type == 'cuda': torch.cuda.synchronize()

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_benchmark_batches: break
            images_dev = images.to(device)
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(images_dev)
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            total_samples_processed += images_dev.size(0)

    if not times:
        print("Недостаточно батчей для бенчмарка инференса.")
        return {"avg_time_per_batch_ms": float('nan'), "throughput_img_per_sec": float('nan')}

    avg_time_per_batch_ms = (sum(times) / len(times)) * 1000
    total_time_sec = sum(times)
    throughput_img_per_sec = total_samples_processed / total_time_sec if total_time_sec > 0 else 0

    print(f"  Инференс: Среднее время на батч: {avg_time_per_batch_ms:.2f} мс")
    print(f"  Инференс: Пропускная способность: {throughput_img_per_sec:.2f} изображений/сек")
    return {"avg_time_per_batch_ms": avg_time_per_batch_ms, "throughput_img_per_sec": throughput_img_per_sec}


def run_training_experiment(model_name, model_class, model_config, train_loader, val_loader, test_loader, epochs,
                            device, compile_model_flag=False):
    print(f"\n--- Эксперимент: {model_name} ---")
    model = model_class(**model_config).to(device)

    if compile_model_flag and device.type == 'cuda':
        print(f"Попытка компиляции модели {model_name} с torch.compile()...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Компиляция для обучения успешна.")
        except Exception as e:
            print(f"Ошибка компиляции {model_name} для обучения: {e}. Обучение без компиляции.")

    num_params = count_parameters(model)
    print(f"Количество обучаемых параметров: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    total_training_time = 0
    peak_gpu_memory_overall = 0
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc, peak_mem_epoch_train = train_model_epoch(model, train_loader, criterion, optimizer,
                                                                        device, scheduler)
        val_loss, val_acc = validate_model_epoch(model, val_loader, criterion, device)
        epoch_duration = time.time() - epoch_start_time
        total_training_time += epoch_duration
        if peak_mem_epoch_train > peak_gpu_memory_overall: peak_gpu_memory_overall = peak_mem_epoch_train
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(
            f"Эпоха {epoch + 1}/{epochs} | Время: {epoch_duration:.2f}с | Train L: {train_loss:.4f}, A: {train_acc:.4f} | Val L: {val_loss:.4f}, A: {val_acc:.4f}")
        if val_acc > best_val_acc: best_val_acc = val_acc

    print("\nОценка на тестовом наборе...")
    test_loss, test_acc = validate_model_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print("\nБенчмарк скорости инференса...")
    inference_stats = benchmark_inference_speed(model, test_loader, device)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Потери ({model_name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Точность ({model_name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    visualize_predictions(model, test_loader, num_samples=10, device=device, model_name=model_name)

    stats = {"model_name": model_name, "params": num_params, "total_train_time_sec": total_training_time,
             "avg_epoch_time_sec": total_training_time / epochs if epochs > 0 else 0,
             "peak_gpu_mem_MB": peak_gpu_memory_overall / (1024 ** 2) if device.type == 'cuda' else "N/A (CPU)",
             "test_accuracy": test_acc, **inference_stats}
    return stats, model


def visualize_predictions(model, test_loader, num_samples=5, device='cpu', model_name="Model"):
    model = model.to(device)
    model.eval()
    try:
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
    except StopIteration:
        print("Не удалось получить данные из test_loader для визуализации.")
        return
    images_subset = images[:num_samples].to(device)
    labels_subset = labels[:num_samples]
    with torch.no_grad():
        outputs = model(images_subset); _, predicted = torch.max(outputs, 1)
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(f"Предсказания модели: {model_name}", fontsize=16)
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i + 1, xticks=[], yticks=[])
        img = images_subset[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray')
        color = "green" if predicted[i] == labels_subset[i] else "red"
        ax.set_title(f"Ист: {labels_subset[i].item()}\nПред: {predicted[i].item()}", color=color)
    plt.show()


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(42))

    batch_size = 128 if device.type == 'cuda' else 32
    num_workers = 2 if device.type == 'cuda' else 0
    pin_memory_flag = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory_flag)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory_flag)

    IMG_SIZE = 28
    PATCH_SIZE = 4
    IN_CHANNELS = 1
    NUM_CLASSES = 10
    DROPOUT = 0.1
    EPOCHS = 5

    if device.type == 'cuda':
        print("Конфигурация ViT для GPU")
        D_MODEL_VIT = 100
        DEPTH_VIT = 4
        NUM_HEADS_VIT = 4
        MLP_RATIO_VIT = 4.0
        # ~493k параметров
    else:
        print("Конфигурация ViT для CPU")
        EPOCHS = 2
        D_MODEL_VIT = 52
        DEPTH_VIT = 2
        NUM_HEADS_VIT = 4
        MLP_RATIO_VIT = 4.0
        # ~70k параметров

    vit_config = {"img_size": IMG_SIZE, "patch_size": PATCH_SIZE, "in_channels": IN_CHANNELS,
                  "num_classes": NUM_CLASSES, "d_model": D_MODEL_VIT, "depth": DEPTH_VIT, "num_heads": NUM_HEADS_VIT,
                  "mlp_ratio": MLP_RATIO_VIT, "dropout": DROPOUT}

    stats_vit, trained_vit_model = run_training_experiment("ViT", VisionTransformer, vit_config, train_loader,
                                                           val_loader, test_loader, EPOCHS, device,
                                                           compile_model_flag=False)

    if device.type == 'cuda': del trained_vit_model; torch.cuda.empty_cache()

    print("\n\n--- Сводка по эксперименту ViT ---")
    metrics_display_names = {"params": "Параметры", "total_train_time_sec": "Общее время обучения",
                             "avg_epoch_time_sec": "Среднее время/эпоха", "peak_gpu_mem_MB": "Пик GPU памяти (обуч.)",
                             "test_accuracy": "Точность на тесте", "avg_time_per_batch_ms": "Инференс (мс/батч)",
                             "throughput_img_per_sec": "Инференс (изобр/сек)"}
    header = f"{'Метрика':<28} | {'ViT':<20}"
    print(header)
    print("-" * (len(header) - 20))

    for key, display_name in metrics_display_names.items():
        if key not in stats_vit: continue
        val_vit = stats_vit[key]
        s_vit = ""
        if key == "params":
            s_vit = f"{val_vit:,}"
        elif key == "test_accuracy":
            s_vit = f"{val_vit:.4f}"
        elif isinstance(val_vit, float):
            s_vit = f"{val_vit:.2f}"
        else:
            s_vit = str(val_vit)
        print(f"{display_name:<28} | {s_vit:<20}")
    print("-" * (len(header) - 20))

    print("\nПримечание: Результаты могут варьироваться.")


if __name__ == "__main__":
    main()
