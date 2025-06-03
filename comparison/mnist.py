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
            out_h = self.C_proj(h)
            outputs.append(out_h)
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
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        mlp_in_val = self.mlp_in(self.norm2(x))
        mlp_hidden, gate = mlp_in_val.chunk(2, dim=-1)
        mlp_activated = mlp_hidden * F.silu(gate)
        x = x + self.dropout_mlp(self.mlp_out(mlp_activated))
        return x


class MambaImageClassifier(nn.Module):
    def __init__(self,
                 img_size=28,
                 in_channels=1,
                 patch_size=4,
                 d_model=64,
                 d_state=16,
                 depth=2,
                 expansion_factor=2,
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
            Reshape((-1, self.num_patches, self.patch_dim)),
            nn.Linear(self.patch_dim, d_model)
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.dropout_embed = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, expansion_factor=expansion_factor, dropout=dropout) for _ in
            range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None: nn.init.zeros_(self.classifier.bias)
        for m in self.patch_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, imgs):
        x = self.patch_proj(imgs)
        x = x + self.pos_embedding
        x = self.dropout_embed(x)
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
        if self.shape[0] == -1:
            return x.reshape(x.shape[0], *self.shape[1:])
        return x.reshape(self.shape)


class ViTBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10,
                 d_model=64, depth=3, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.depth = depth
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_proj = nn.Sequential(
            Reshape((-1, in_channels, img_size, img_size)),
            nn.Unfold(kernel_size=patch_size, stride=patch_size),
            Reshape((-1, num_patches, patch_dim)),
            nn.Linear(patch_dim, d_model)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.dropout_embed = nn.Dropout(dropout)
        self.transformer_encoder = nn.ModuleList([
            ViTBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
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
        for blk in self.transformer_encoder:
            x = blk(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = torch.cuda.memory_allocated(device)

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
    if device.type == 'cuda':
        peak_memory_epoch = torch.cuda.max_memory_allocated(device) - initial_memory
    return avg_loss, accuracy, peak_memory_epoch


def validate_model_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Валидация", leave=False, ncols=100):
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


def run_training_experiment(model_name, model, train_loader, val_loader, test_loader, epochs, device):
    print(f"\n--- Обучение модели: {model_name} ---")
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Количество обучаемых параметров: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    total_training_time = 0
    peak_gpu_memory_overall = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

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
        print(f"Эпоха {epoch + 1}/{epochs} | Время: {epoch_duration:.2f}с | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # torch.save(model.state_dict(), f'best_{model_name.lower().replace(" ", "_")}_model.pth')

    print("\nОценка на тестовом наборе...")
    test_loss, test_acc = validate_model_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Потери ({model_name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Точность ({model_name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.tight_layout()
    plt.show()
    visualize_predictions(model, test_loader, num_samples=10, device=device, model_name=model_name)
    stats = {
        "model_name": model_name, "params": num_params, "total_train_time_sec": total_training_time,
        "avg_epoch_time_sec": total_training_time / epochs if epochs > 0 else 0,
        "peak_gpu_mem_MB": peak_gpu_memory_overall / (1024 ** 2) if device.type == 'cuda' else "N/A (CPU)",
        "test_accuracy": test_acc
    }
    return stats


def visualize_predictions(model, test_loader, num_samples=5, device='cuda' if torch.cuda.is_available() else 'cpu',
                          model_name="Model"):
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
        outputs = model(images_subset)
        _, predicted = torch.max(outputs, 1)
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

    # параметры для моделей подобраны для сопоставимого числа параметров
    if device.type == 'cuda':
        print("Конфигурация для GPU (цель: ~490k параметров)")
        # Mamba (GPU)
        D_MODEL_MAMBA = 128
        DEPTH_MAMBA = 4
        D_STATE_MAMBA = 16
        EXP_FACTOR_MAMBA = 2
        # ViT (GPU)
        D_MODEL_VIT = 100
        DEPTH_VIT = 4
        NUM_HEADS_VIT = 4
        MLP_RATIO_VIT = 4.0
    else:  # CPU
        print("Конфигурация для CPU (цель: ~65-70k параметров)")
        EPOCHS = 2
        # Mamba (CPU)
        D_MODEL_MAMBA = 64
        DEPTH_MAMBA = 2
        D_STATE_MAMBA = 16
        EXP_FACTOR_MAMBA = 2
        # ViT (CPU)
        D_MODEL_VIT = 52
        DEPTH_VIT = 2
        NUM_HEADS_VIT = 4
        MLP_RATIO_VIT = 4.0

    mamba_model_config = {
        "img_size": IMG_SIZE, "in_channels": IN_CHANNELS, "patch_size": PATCH_SIZE,
        "d_model": D_MODEL_MAMBA, "d_state": D_STATE_MAMBA, "depth": DEPTH_MAMBA,
        "expansion_factor": EXP_FACTOR_MAMBA, "num_classes": NUM_CLASSES, "dropout": DROPOUT
    }
    mamba_model = MambaImageClassifier(**mamba_model_config)

    vit_model_config = {
        "img_size": IMG_SIZE, "patch_size": PATCH_SIZE, "in_channels": IN_CHANNELS,
        "num_classes": NUM_CLASSES, "d_model": D_MODEL_VIT, "depth": DEPTH_VIT,
        "num_heads": NUM_HEADS_VIT, "mlp_ratio": MLP_RATIO_VIT, "dropout": DROPOUT
    }
    vit_model = VisionTransformer(**vit_model_config)

    all_stats = []
    stats_mamba = run_training_experiment("Mamba", mamba_model, train_loader, val_loader, test_loader, EPOCHS, device)
    all_stats.append(stats_mamba)

    if device.type == 'cuda':
        del mamba_model
        torch.cuda.empty_cache()
        print("\nОчищена память GPU перед обучением ViT.\n")
        time.sleep(1)

    stats_vit = run_training_experiment("ViT", vit_model, train_loader, val_loader, test_loader, EPOCHS, device)
    all_stats.append(stats_vit)

    if device.type == 'cuda':
        del vit_model
        torch.cuda.empty_cache()

    print("\n\n--- Сводка по экспериментам ---")
    header = f"{'Метрика':<25} | {all_stats[0]['model_name']:<20} | {all_stats[1]['model_name']:<20}"
    print(header)
    print("-" * len(header))
    for key in ["params", "total_train_time_sec", "avg_epoch_time_sec", "peak_gpu_mem_MB", "test_accuracy"]:
        val_mamba = all_stats[0][key]
        val_vit = all_stats[1][key]

        if key == "params":
            s_mamba = f"{val_mamba:,}"
            s_vit = f"{val_vit:,}"
        elif isinstance(val_mamba, float) and key != "test_accuracy":
            s_mamba = f"{val_mamba:.2f}"
            s_vit = f"{val_vit:.2f}"
            if key == "total_train_time_sec" or key == "avg_epoch_time_sec":
                s_mamba += " с"
                s_vit += " с"
            elif key == "peak_gpu_mem_MB":
                s_mamba += " МБ"
                s_vit += " МБ"
        elif key == "test_accuracy":
            s_mamba = f"{val_mamba:.4f}"
            s_vit = f"{val_vit:.4f}"
        else:
            s_mamba = str(val_mamba)
            s_vit = str(val_vit)

        metric_name_display = {
            "params": "Параметры",
            "total_train_time_sec": "Общее время обучения",
            "avg_epoch_time_sec": "Среднее время/эпоха",
            "peak_gpu_mem_MB": "Пик GPU памяти (обуч.)",
            "test_accuracy": "Точность на тесте"
        }.get(key, key)
        print(f"{metric_name_display:<25} | {s_mamba:<20} | {s_vit:<20}")
    print("-" * len(header))

    print("\nАнализ результатов:")
    time_comparison = "быстрее" if all_stats[0]['total_train_time_sec'] < all_stats[1]['total_train_time_sec'] else \
        "медленнее" if all_stats[0]['total_train_time_sec'] > all_stats[1][
            'total_train_time_sec'] else "сравнимо по времени с"
    print(f"- Mamba обучалась {time_comparison} ViT.")

    if device.type == 'cuda':
        mem_comparison = "меньше" if all_stats[0]['peak_gpu_mem_MB'] < all_stats[1]['peak_gpu_mem_MB'] else \
            "больше" if all_stats[0]['peak_gpu_mem_MB'] > all_stats[1]['peak_gpu_mem_MB'] else "сравнимо по"
        print(f"- Mamba потребляла {mem_comparison} пиковой GPU памяти во время обучения, чем ViT.")
    else:
        print("- Сравнение потребления GPU памяти недоступно (используется CPU).")

    acc_comparison = "лучшую" if all_stats[0]['test_accuracy'] > all_stats[1]['test_accuracy'] else \
        "худшую" if all_stats[0]['test_accuracy'] < all_stats[1]['test_accuracy'] else "сравнимую"
    print(f"- Mamba показала {acc_comparison} точность на тестовом наборе по сравнению с ViT.")

    param_diff_percent = abs(all_stats[0]['params'] - all_stats[1]['params']) / all_stats[0]['params'] * 100
    print(f"- Разница в количестве параметров: {param_diff_percent:.2f}%")

    print("\nПримечание: Результаты могут варьироваться. Для более надежного сравнения "
          "может потребоваться больше запусков и/или более длительное обучение.")


if __name__ == "__main__":
    main()
