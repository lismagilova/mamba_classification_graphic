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
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        if self.shape[0] == -1: return x.reshape(x.shape[0], *self.shape[1:])
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
        A_exp = torch.exp(self.A.unsqueeze(0).unsqueeze(0) * delta)
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = A_exp[:, t] * h + B[:, t]
            out_t = self.C_proj(h)
            outputs.append(out_t)
        outputs = torch.stack(outputs, dim=1)
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
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        mlp_in_val = self.mlp_in(self.norm2(x))
        mlp_hidden, gate = mlp_in_val.chunk(2, dim=-1)
        mlp_activated = mlp_hidden * F.silu(gate)
        x = x + self.dropout_mlp(self.mlp_out(mlp_activated))
        return x


class MambaImageClassifier(nn.Module):
    def __init__(self, img_size=32, in_channels=3, patch_size=4, d_model=128,
                 d_state=16, depth=4, expansion_factor=2,
                 num_classes=10, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_proj = nn.Sequential(
            Reshape((-1, in_channels, img_size, img_size)),
            nn.Unfold(kernel_size=patch_size, stride=patch_size),  # (B, C*P*P, N_patches)
            PermuteAndReshape((-1, self.patch_dim, self.num_patches), (0, 2, 1)),  # (B, N_patches, C*P*P)
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
        if isinstance(self.classifier, nn.Linear):
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
        for block in self.blocks: x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


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
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 d_model=128, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.depth = depth
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_proj = nn.Sequential(
            Reshape((-1, in_channels, img_size, img_size)),
            nn.Unfold(kernel_size=patch_size, stride=patch_size),  # (B, C*P*P, N_patches)
            PermuteAndReshape((-1, patch_dim, num_patches), (0, 2, 1)),  # (B, N_patches, C*P*P)
            nn.Linear(patch_dim, d_model)
        )
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


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def unnormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    if not times: return {"avg_time_per_batch_ms": float('nan'), "throughput_img_per_sec": float('nan')}
    avg_time_per_batch_ms = (sum(times) / len(times)) * 1000
    total_time_sec = sum(times)
    throughput_img_per_sec = total_samples_processed / total_time_sec if total_time_sec > 0 else 0
    print(
        f"  Инференс: Среднее время на батч: {avg_time_per_batch_ms:.2f} мс, Пропускная способность: {throughput_img_per_sec:.2f} изобр/сек")
    return {"avg_time_per_batch_ms": avg_time_per_batch_ms, "throughput_img_per_sec": throughput_img_per_sec}


def run_training_experiment(model_name, model_class, model_config, train_loader, val_loader, test_loader, epochs,
                            device, compile_model_flag=False):
    print(f"\n--- Эксперимент: {model_name} ---")
    model = model_class(**model_config).to(device)
    if compile_model_flag and device.type == 'cuda':
        print(f"Попытка компиляции {model_name}...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Компиляция успешна.")
        except Exception as e:
            print(f"Ошибка компиляции: {e}. Обучение без компиляции.")
    num_params = count_parameters(model)
    print(f"Параметры: {num_params:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
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
            f"Эпоха {epoch + 1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.2e} | Время: {epoch_duration:.2f}с | Train L: {train_loss:.4f}, A: {train_acc:.4f} | Val L: {val_loss:.4f}, A: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),f'best_{model_name.lower()}_cifar10.pth')
            print(f"  Лучшая модель сохранена: Val Acc {best_val_acc:.4f}")
    print("\nОценка на тестовом наборе (лучшая модель)...")
    model.load_state_dict(torch.load(f'best_{model_name.lower()}_cifar10.pth'))
    test_loss, test_acc = validate_model_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print("\nБенчмарк скорости инференса (лучшая модель)...")
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
        outputs = model(images_subset)
        _, predicted = torch.max(outputs, 1)
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Предсказания модели: {model_name}", fontsize=16)
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i + 1, xticks=[], yticks=[])
        img = images_subset[i].cpu()
        img = unnormalize(img, CIFAR_MEAN, CIFAR_STD)
        img = img.permute(1, 2, 0)
        img = torch.clamp(img, 0, 1)
        ax.imshow(img.numpy())
        true_label = cifar10_classes[labels_subset[i]]
        pred_label = cifar10_classes[predicted[i].item()]
        color = "green" if predicted[i] == labels_subset[i] else "red"
        ax.set_title(f"Ист: {true_label}\nПред: {pred_label}", color=color, fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(42))

    batch_size = 128 if device.type == 'cuda' else 32
    num_workers = 2 if device.type == 'cuda' else 0
    pin_memory_flag = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory_flag)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory_flag)

    IMG_SIZE = 32
    PATCH_SIZE = 4
    IN_CHANNELS = 3
    NUM_CLASSES = 10
    DROPOUT = 0.1
    EPOCHS = 20

    COMPILE_MAMBA_FOR_TRAINING = False

    D_MODEL_MAMBA = 128
    DEPTH_MAMBA = 4
    D_STATE_MAMBA = 16
    EXP_FACTOR_MAMBA = 2

    D_MODEL_VIT = 128
    DEPTH_VIT = 4
    NUM_HEADS_VIT = 4
    MLP_RATIO_VIT = 2.0

    if device.type == 'cpu':
        EPOCHS = 5
        D_MODEL_MAMBA = 64
        DEPTH_MAMBA = 2
        D_STATE_MAMBA = 16
        EXP_FACTOR_MAMBA = 2
        D_MODEL_VIT = 64
        DEPTH_VIT = 2
        NUM_HEADS_VIT = 2
        MLP_RATIO_VIT = 2.0
        COMPILE_MAMBA_FOR_TRAINING = False

    mamba_config = {"img_size": IMG_SIZE, "in_channels": IN_CHANNELS, "patch_size": PATCH_SIZE,
                    "d_model": D_MODEL_MAMBA, "d_state": D_STATE_MAMBA, "depth": DEPTH_MAMBA,
                    "expansion_factor": EXP_FACTOR_MAMBA, "num_classes": NUM_CLASSES, "dropout": DROPOUT}
    vit_config = {"img_size": IMG_SIZE, "patch_size": PATCH_SIZE, "in_channels": IN_CHANNELS,
                  "num_classes": NUM_CLASSES, "d_model": D_MODEL_VIT, "depth": DEPTH_VIT, "num_heads": NUM_HEADS_VIT,
                  "mlp_ratio": MLP_RATIO_VIT, "dropout": DROPOUT}

    all_stats_data = []

    stats_mamba, trained_mamba_model = run_training_experiment("Mamba", MambaImageClassifier, mamba_config,
                                                               train_loader, val_loader, test_loader, EPOCHS, device,
                                                               compile_model_flag=COMPILE_MAMBA_FOR_TRAINING)
    all_stats_data.append(stats_mamba)
    if device.type == 'cuda':
        del trained_mamba_model
        torch.cuda.empty_cache()
        print("\nОчищена память GPU.\n")
        time.sleep(0.5)

    stats_vit, trained_vit_model = run_training_experiment("ViT", VisionTransformer, vit_config, train_loader,
                                                           val_loader, test_loader, EPOCHS, device,
                                                           compile_model_flag=False)
    all_stats_data.append(stats_vit)
    if device.type == 'cuda':
        del trained_vit_model
        torch.cuda.empty_cache()

    print("\n\n--- Сводка по экспериментам (CIFAR-10) ---")
    metrics_display_names = {"params": "Параметры", "total_train_time_sec": "Общее время обучения",
                             "avg_epoch_time_sec": "Среднее время/эпоха", "peak_gpu_mem_MB": "Пик GPU памяти (обуч.)",
                             "test_accuracy": "Точность на тесте", "avg_time_per_batch_ms": "Инференс (мс/батч)",
                             "throughput_img_per_sec": "Инференс (изобр/сек)"}
    header = f"{'Метрика':<28} | {all_stats_data[0]['model_name']:<20} | {all_stats_data[1]['model_name']:<20}"
    print(header)
    print("-" * len(header))
    for key in metrics_display_names.keys():
        if key not in all_stats_data[0] or key not in all_stats_data[1]: continue
        val_mamba, val_vit = all_stats_data[0][key], all_stats_data[1][key]
        s_mamba, s_vit = "", ""
        if key == "params":
            s_mamba, s_vit = f"{val_mamba:,}", f"{val_vit:,}"
        elif key == "test_accuracy":
            s_mamba, s_vit = f"{val_mamba:.4f}", f"{val_vit:.4f}"
        elif isinstance(val_mamba, float):
            s_mamba, s_vit = f"{val_mamba:.2f}", f"{val_vit:.2f}"
        else:
            s_mamba, s_vit = str(val_mamba), str(val_vit)
        print(f"{metrics_display_names[key]:<28} | {s_mamba:<20} | {s_vit:<20}")
    print("-" * len(header))

    model_names = [s['model_name'] for s in all_stats_data]
    inference_times = [s.get('avg_time_per_batch_ms', float('nan')) for s in all_stats_data]
    throughputs = [s.get('throughput_img_per_sec', float('nan')) for s in all_stats_data]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Сравнение скорости инференса (CIFAR-10)", fontsize=16)
    axs[0].bar(model_names, inference_times, color=['skyblue', 'lightcoral'])
    axs[0].set_title('Среднее время на батч (мс)')
    axs[0].set_ylabel('Время (мс)')
    axs[0].grid(axis='y', linestyle='--')
    for i, v in enumerate(inference_times): axs[0].text(i, v + np.nan_to_num(v) * 0.01 if not np.isnan(v) else 0,
                                                        f"{v:.2f}", ha='center', va='bottom')
    axs[1].bar(model_names, throughputs, color=['skyblue', 'lightcoral'])
    axs[1].set_title('Пропускная способность (изобр/сек)')
    axs[1].set_ylabel('Изображений / сек')
    axs[1].grid(axis='y', linestyle='--')
    for i, v in enumerate(throughputs): axs[1].text(i, v + np.nan_to_num(v) * 0.01 if not np.isnan(v) else 0,
                                                    f"{v:.2f}", ha='center', va='bottom')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\nАнализ результатов (CIFAR-10):")
    param_mamba = all_stats_data[0]['params']
    param_vit = all_stats_data[1]['params']
    param_diff_percent = (abs(param_mamba - param_vit) / max(1, param_mamba)) * 100
    print(f"- Разница в параметрах: {param_diff_percent:.2f}% ({param_mamba:,} vs {param_vit:,})")
    t_m, t_v = all_stats_data[0]['total_train_time_sec'], all_stats_data[1]['total_train_time_sec']
    print(f"- {'Mamba' if t_m < t_v else 'ViT'} быстрее обучилась ({min(t_m, t_v):.2f}с vs {max(t_m, t_v):.2f}с).")
    if device.type == 'cuda':
        m_m_val, m_v_val = all_stats_data[0]['peak_gpu_mem_MB'], all_stats_data[1]['peak_gpu_mem_MB']
        if isinstance(m_m_val, float) and isinstance(m_v_val, float):
            print(
                f"- {'Mamba' if m_m_val < m_v_val else 'ViT'} меньше GPU памяти ({min(m_m_val, m_v_val):.2f}МБ vs {max(m_m_val, m_v_val):.2f}МБ).")
        else:
            print("- GPU память: некорректные данные.")
    else:
        print("- GPU память: сравнение недоступно (CPU).")
    a_m, a_v = all_stats_data[0]['test_accuracy'], all_stats_data[1]['test_accuracy']
    if abs(a_m - a_v) < 1e-3:
        print(f"- Точность сравнима (~{a_m:.4f}).")
    else:
        print(f"- {'Mamba' if a_m > a_v else 'ViT'} лучше точность ({max(a_m, a_v):.4f} vs {min(a_m, a_v):.4f}).")
    it_m_val, it_v_val = all_stats_data[0].get('avg_time_per_batch_ms', float('inf')), all_stats_data[1].get(
        'avg_time_per_batch_ms', float('inf'))
    if not any(map(lambda x: math.isnan(x) or math.isinf(x), [it_m_val, it_v_val])):
        print(
            f"- {'Mamba' if it_m_val < it_v_val else 'ViT'} быстрее инференс ({min(it_m_val, it_v_val):.2f} мс/батч vs {max(it_m_val, it_v_val):.2f} мс/батч).")
    else:
        print("- Инференс: некорректные данные.")
    print("\nПримечание: Результаты могут варьироваться. `torch.compile` для Mamba было отключено.")


if __name__ == "__main__":
    main()
