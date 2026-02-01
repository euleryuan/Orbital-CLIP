"""
---------------------------------------------------
Implementation of the fine-tuning pipeline for Multimodal Large Language Models (specifically CLIP)
focused on Rare Orbital Disease Detection.

This script manages:
1. Clinical-specific prompt engineering and augmentation.
2. Contrastive loss optimization (OpenCLIP backbone).
3. Zero-shot validation protocols using distinct clinical terminology.

"""

import os
import random
import logging
import argparse
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip

# --- System Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Prompt Engineering Registry ---
class ClinicalPromptRegistry:
    """
    Central registry for managing clinical textual prompts.
    Implements the prompt augmentation strategy defined in the manuscript (v4).
    """
    # Dataset folder names (Source of Truth for loading data)
    DATASET_CLASSES = ["orbital disease", "healthy eyes", "non-orbital disease"]
    
    # Specific class names used for Zero-Shot Validation (Note: 'no eye disease' differs from folder name)
    VALIDATION_CLASSES = ("orbital disease", "no eye disease", "non-orbital disease")

    # Prompt Dictionary (v4 Protocols)
    _PROMPTS: Dict[str, List[str]] = {
        DATASET_CLASSES[0]: [  # Orbital Disease
            "This medical image shows a single eye exhibiting clear signs of orbital disease, such as inflammation, asymmetry, or unusual bulging.",
            "In this image, the visible eye displays characteristics of orbital disease, including puffiness, discoloration, or abnormal protrusion.",
            "The single eye in this diagnostic image is marked by signs of orbital disease, noticeable through abnormalities like swelling or structural changes around the socket.",
            "This is an image of an individual with orbital disease in the eye region.",
            "This image shows an eye diagnosed with an orbital disorder.",
            "orbital eye disease",
        ],
        DATASET_CLASSES[1]: [  # Healthy Eyes
            "no eye disease",
            "free from eye disease",
            "This is an image showing healthy eye."
        ],
        DATASET_CLASSES[2]: [  # Non-Orbital Disease
            "This image depicts a single eye from an individual suffering from a non-orbital disease, showing symptoms unrelated to the eye or orbital area.",
            "The single eye in this photograph belongs to a person affected by a non-orbital disease, with evident health issues that do not involve the eye region.",
            "In this diagnostic image, the depicted symptoms are of a non-orbital disease, impacting areas of the body other than the orbital region, with the single eye shown being unrelated.",
            "non-orbital eye disease",
        ]
    }

    @classmethod
    def sample_prompt(cls, class_idx: int) -> str:
        """Randomly samples a prompt for training augmentation."""
        class_name = cls.DATASET_CLASSES[class_idx]
        return random.choice(cls._PROMPTS[class_name])

# --- Dataset Implementation ---
class OrbitalMultimodalDataset(Dataset):
    def __init__(self, root_dir: str, preprocess_fn):
        """
        Custom Dataset handler for Orbital Imaging data.
        
        Args:
            root_dir (str): Path to the dataset root.
            preprocess_fn: CLIP-specific image transformation pipeline.
        """
        self.root_dir = root_dir
        self.preprocess = preprocess_fn
        self.classes = ClinicalPromptRegistry.DATASET_CLASSES
        self.image_paths: List[str] = []
        self.targets: List[int] = []
        
        self._index_dataset()

    def _index_dataset(self):
        """Traverses directories to index valid image files."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                logger.warning(f"Class directory missing: {class_dir}")
                continue

            # Filter valid image extensions
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_exts)]
            
            for img_name in images:
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.targets.append(idx)
        
        logger.info(f"Indexed {len(self.image_paths)} images from {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.targets[idx]

        # Robust image loading
        try:
            image = Image.open(img_path).convert("RGB")
            if self.preprocess:
                image = self.preprocess(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank tensor in case of corruption (handling edge cases)
            return torch.zeros(3, 224, 224), label, "Error"

        # Prompt Sampling Strategy
        caption = ClinicalPromptRegistry.sample_prompt(label)
        
        return image, label, caption

# --- Training Engine ---
def execute_training_protocol(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args
):
    """
    Executes the main training loop with Sequential Sensitivity Analysis monitoring.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer configuration (Strict adherence to original parameters)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    
    # Loss function instantiation
    loss_fn = open_clip.create_loss(args)
    
    best_accuracy = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    logger.info(f"Starting training on {device} for {args.epochs} epochs.")

    for epoch in range(args.epochs):
        # ================= Training Phase =================
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)
        
        for images, labels, captions in pbar:
            images = images.to(device)
            # labels are unused in contrastive loss but kept for structure
            
            # Tokenize dynamic prompts
            text_tokens = open_clip.tokenize(captions).to(device)

            # Feature Encoding
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            # Manual Normalization (Preserving original logic)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Logit Computation
            logit_scale = model.logit_scale
            
            # Compute OpenCLIP Contrastive Loss
            loss = loss_fn(image_features, text_features, logit_scale)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} | Training Loss: {avg_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Checkpoint interval
        if (epoch + 1) % args.save_interval != 0:
            continue

        # ================= Validation Phase =================
        model.eval()
        correct_top1 = 0
        total_samples = 0
        total_val_loss = 0.0
        
        # Prepare fixed validation prompts
        val_prompts = ClinicalPromptRegistry.VALIDATION_CLASSES
        val_text_tokens = open_clip.tokenize(val_prompts).to(device)

        with torch.no_grad():
            # Pre-compute text features for validation (Optimization)
            val_text_features = model.encode_text(val_text_tokens)
            val_text_features = val_text_features / val_text_features.norm(dim=1, keepdim=True)

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False)
            
            for images, labels, _ in val_pbar:
                images = images.to(device)
                labels = labels.to(device)

                # Image Encoding
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                # Similarity Calculation (Logic: Scale * (Img @ Text.T))
                logit_scale = model.logit_scale
                logits_per_image = logit_scale * (image_features @ val_text_features.T)

                # Accuracy Calculation
                probs = logits_per_image.softmax(dim=-1)
                top1_pred = torch.argsort(probs, dim=-1, descending=True)[:, 0]
                
                correct_top1 += (top1_pred == labels).sum().item()
                total_samples += labels.size(0)

        # Metrics
        accuracy = correct_top1 / total_samples
        logger.info(f"Epoch {epoch + 1} | Val Accuracy: {accuracy:.4f}")

        # Best Model Preservation
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = os.path.join(args.save_dir, "clip_model_epoch_best.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New Best Model Saved! Accuracy: {best_accuracy:.4f}")

    # Final Save
    final_path = os.path.join(args.save_dir, "clip_model_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training protocol completed. Final Best Accuracy: {best_accuracy:.4f}")

# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orbital Disease Detection - CLIP Fine-tuning")
    
    # Model Configuration (Defaults set to original script values)
    parser.add_argument("--model-name", type=str, default="ViT-B-32-256")
    parser.add_argument("--pretrained", type=str, default="clip_model_checkpoints/clip_model_epoch_best-9021.pt")
    parser.add_argument("--cache-dir", type=str, default="model_download/")
    
    # Data Configuration
    parser.add_argument("--train-dir", type=str, default="dataset_orbital_diseases/train")
    parser.add_argument("--val-dir", type=str, default="dataset_orbital_diseases/val")
    parser.add_argument("--save-dir", type=str, default="clip_model_checkpoints")
    
    # Training Hyperparameters
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size (Default: 200)")
    parser.add_argument("--lr", type=float, default=1e-8, help="Learning rate (Default: 1e-8)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1)
    
    # OpenCLIP Distributed Args (Required for loss function initialization)
    parser.add_argument("--local-loss", action="store_true", default=False)
    parser.add_argument("--gather-with-grad", action="store_true", default=False)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--horovod", action="store_true", default=False)
    parser.add_argument("--distill", type=int, default=0)
    parser.add_argument("--siglip", default=False)

    args = parser.parse_args()

    # Model Initialization
    logger.info(f"Initializing model: {args.model_name}")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        cache_dir=args.cache_dir
    )

    # Data Loaders
    logger.info("Initializing data loaders...")
    train_dataset = OrbitalMultimodalDataset(args.train_dir, preprocess_train)
    val_dataset = OrbitalMultimodalDataset(args.val_dir, preprocess_val)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # Execute
    execute_training_protocol(model, train_loader, val_loader, args)
