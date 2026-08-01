import os, gc, random, numpy as np, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from google.colab import drive

# ── Mount Drive for safe checkpointing ──
drive.mount('/drive', force_remount=True)
DRIVE_DIR = "/drive/MyDrive/brain_tumor_checkpoints"
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs("/content/checkpoints", exist_ok=True)
os.makedirs("/content/results", exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Device : {device}")
if device == "cuda":
    print(f"✓ GPU    : {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

IMG_MEAN    = [0.485, 0.456, 0.406]
IMG_STD     = [0.229, 0.224, 0.225]
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ── Dataset path ──
train_dir = "/content/data/Training"
test_dir  = "/content/data/Testing"

for split, d in [("Train", train_dir), ("Test", test_dir)]:
    for cls in sorted(os.listdir(d)):
        n = len(os.listdir(os.path.join(d, cls)))
        print(f"  {split}/{cls}: {n}")

# ── Transforms ──
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(test_dir,  transform=val_transform)

# Glioma 5x oversampling
custom_weights = np.array([5.0, 1.0, 1.0, 1.0])
sample_weights = [custom_weights[label] for _, label in train_dataset.samples]
sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"\n✓ Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# Glioma 5x loss penalty
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([5.0, 1.0, 1.0, 1.0]).to(device),
    label_smoothing=0.05
)

# ── Model: ConvNeXt-Base ──
class ConvNeXtBrain(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        self.backbone = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        in_f = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_f, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


model = ConvNeXtBrain().to(device)
print(f"✓ ConvNeXt-Base | Params: {sum(p.numel() for p in model.parameters()):,}")


# ── Evaluate ──
@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0
    all_p, all_l = [], []
    for imgs, labels in tqdm(val_loader, desc="  val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        probs = torch.softmax(model(imgs), 1)
        correct += (probs.argmax(1) == labels).sum().item()
        total   += labels.size(0)
        all_p.extend(probs.argmax(1).cpu().numpy())
        all_l.extend(labels.cpu().numpy())
    return correct / total, np.array(all_p), np.array(all_l)


# ── TTA Evaluate ──
@torch.no_grad()
def evaluate_tta():
    model.eval()
    tta_tf = [
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation((90, 90)), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation((180, 180)), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]),
        transforms.Compose([transforms.Resize((240, 240)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation((-90, -90)), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.15, contrast=0.15), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]),
    ]
    all_probs  = None
    all_labels = None
    for i, tf in enumerate(tta_tf):
        print(f"  TTA {i+1}/{len(tta_tf)}...")
        ds     = datasets.ImageFolder(test_dir, transform=tf)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
        probs_list, labels_list = [], []
        for imgs, labels in loader:
            imgs  = imgs.to(device)
            probs = torch.softmax(model(imgs), 1)
            probs_list.extend(probs.cpu().numpy())
            labels_list.extend(labels.numpy())
        pa = np.array(probs_list)
        if all_probs is None:
            all_probs  = pa
            all_labels = np.array(labels_list)
        else:
            all_probs += pa
    all_probs /= len(tta_tf)
    preds = all_probs.argmax(axis=1)
    return (preds == all_labels).mean(), preds, all_labels


def save_checkpoint(name, state, acc):
    local = f"/content/checkpoints/{name}.pth"
    drive = f"{DRIVE_DIR}/{name}.pth"
    torch.save(state, local)
    torch.save(state, drive)
    print(f"  ✓ Saved {name} ({acc*100:.4f}%) → Drive + Local")


# ── Phase runner ──
def run_phase(name, epochs, max_lr, best_acc, use_onecycle=True):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}\n{name}\nTrainable: {trainable:,}\n{'='*60}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=max_lr / 10, weight_decay=1e-4
    )
    if use_onecycle:
        scheduler = OneCycleLR(optimizer, max_lr=max_lr,
                               steps_per_epoch=len(train_loader),
                               epochs=epochs, pct_start=0.2)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-8)

    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc="  train", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if use_onecycle:
                scheduler.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        if not use_onecycle:
            scheduler.step()

        va, preds, labels_arr = evaluate()
        print(f"[Ep {ep:02d}/{epochs}] Train:{correct/total*100:.2f}%  Val:{va*100:.4f}%", end="")

        if va > best_acc:
            best_acc   = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_checkpoint(f"best_{name.split()[0].lower()}", best_state, va)
            print(f"  ✓ NEW BEST!")
        else:
            print()

        gc.collect()
        torch.cuda.empty_cache()

    if best_state:
        model.load_state_dict(best_state)
    return best_acc


# ════════════════════════════════════════════════════════
# PHASE A — Frozen backbone, head only (10 epochs)
# ════════════════════════════════════════════════════════
for p in model.backbone.features.parameters():
    p.requires_grad = False
best = run_phase("PHASE A Head only", epochs=10, max_lr=3e-4, best_acc=0.0)

# ════════════════════════════════════════════════════════
# PHASE B — Unfreeze top 2 stages (25 epochs)
# ════════════════════════════════════════════════════════
for stage in list(model.backbone.features.children())[-2:]:
    for p in stage.parameters():
        p.requires_grad = True
best = run_phase("PHASE B Top2", epochs=25, max_lr=1e-4, best_acc=best)

# ════════════════════════════════════════════════════════
# PHASE C — Full model fine-tune (20 epochs)
# ════════════════════════════════════════════════════════
for p in model.parameters():
    p.requires_grad = True
best = run_phase("PHASE C Full", epochs=20, max_lr=2e-5, best_acc=best)

# ════════════════════════════════════════════════════════
# PHASE D — Deep glioma-focused fine-tune (30 epochs)
# ════════════════════════════════════════════════════════
best = run_phase("PHASE D Glioma", epochs=30, max_lr=5e-6,
                 best_acc=best, use_onecycle=False)

# ════════════════════════════════════════════════════════
# FINAL — TTA Evaluation
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL TTA EVALUATION (8 passes)")
print("="*60)
tta_acc, tta_preds, tta_labels = evaluate_tta()
print(f"\n{'='*60}")
print(f"FINAL TTA ACCURACY: {tta_acc*100:.4f}%")
print("="*60)
print(classification_report(tta_labels, tta_preds, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(tta_labels, tta_preds)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Brain Tumor Classification\nTTA Accuracy: {tta_acc*100:.2f}%",
          fontsize=13, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("/content/results/confusion_matrix.png", dpi=150)
plt.savefig(f"{DRIVE_DIR}/confusion_matrix.png", dpi=150)
plt.show()
print(f"\n✓ Confusion matrix saved to Drive!")
print(f"✓ Best model saved to Drive: {DRIVE_DIR}/")
print(f"✓ Best accuracy: {best*100:.4f}%")
print(f"✓ TTA accuracy:  {tta_acc*100:.4f}%")
