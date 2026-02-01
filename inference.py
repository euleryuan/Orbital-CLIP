"""
External Validation & Inference Engine
--------------------------------------
Performs batch inference on external validation cohorts to assess the generalization 
capability of the fine-tuned Orbital-CLIP model.

Metrics: Confusion Matrix, Classification Report (Precision/Recall/F1), Top-1 Accuracy.
"""

import os
import argparse
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceConfig:

    CLASS_NAMES = ["orbital disease", "healthy eyes", "non-orbital disease"]
    

    # Context Length = 77
    PROMPTS = [
        "orbital disease",
        "no eye disease",
        "non-orbital disease."  
    ]
    
    CONTEXT_LENGTH = 77
    BATCH_SIZE = 140

class DataIngestion:
    """Handles the loading of external validation cohorts."""
    
    @staticmethod
    def load_cohort(dataset_root: str) -> Tuple[List[str], pd.DataFrame]:
        """
        Scans the directory structure to build the inference manifest.
        Expected structure: root/class_name/image.jpg
        """
        logger.info(f"Scanning external cohort at: {dataset_root}")
        
        image_paths = []
        labels_true = []
        indices = []

        
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset path {dataset_root} does not exist.")

        
        subfolders = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        
        for folder_name in subfolders:

            
            folder_path = os.path.join(dataset_root, folder_name)

            current_label = folder_name 
            
            
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(folder_path, img_file)
                    image_paths.append(full_path)
                    indices.append(img_file) 
                    labels_true.append(current_label) 

        df = pd.DataFrame({
            'image_index': indices,
            'true_label': labels_true,
            'file_path': image_paths
        })
        df.set_index('image_index', inplace=True)
        
        logger.info(f"Cohort loaded. Total samples: {len(df)}")
        return image_paths, df

class ModelInference:
    def __init__(self, model_name: str, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model backbone: {model_name} on {self.device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=checkpoint_path
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def run_batch_inference(self, image_paths: List[str], class_prompts: List[str]):
        """Executes batched inference."""
        
        # Tokenize Prompts
        text_tokens = self.tokenizer(class_prompts, context_length=InferenceConfig.CONTEXT_LENGTH).to(self.device)
        
        all_logits = []
        all_indices = []
        
        total_images = len(image_paths)
        num_batches = (total_images + InferenceConfig.BATCH_SIZE - 1) // InferenceConfig.BATCH_SIZE
        
        logger.info(f"Starting inference: {total_images} images in {num_batches} batches.")

        with torch.no_grad():
            # Pre-compute text features (Efficiency optimization)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            for i in tqdm(range(num_batches), desc="Inferencing"):
                batch_paths = image_paths[i * InferenceConfig.BATCH_SIZE : (i + 1) * InferenceConfig.BATCH_SIZE]
                
                # Load and Preprocess Batch
                batch_tensors = []
                valid_batch_indices = [] # Keep track of indices that successfully loaded
                
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        batch_tensors.append(self.preprocess(img))
                        valid_batch_indices.append(os.path.basename(path)) # Use filename as index
                    except Exception as e:
                        logger.error(f"Failed to load {path}: {e}")
                        continue
                
                if not batch_tensors:
                    continue

                batch_input = torch.stack(batch_tensors).to(self.device)
                
                # Image Encoding
                image_features = self.model.encode_image(batch_input)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
                # Logit Computation (Logic: Scale * (Img @ Text.T))
                logit_scale = self.model.logit_scale 
                logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                
                all_logits.append(logits.cpu().numpy())
                
        return np.concatenate(all_logits, axis=0)

class EvaluationEngine:
    @staticmethod
    def generate_report(y_true: List[str], y_pred: List[str], classes: List[str], output_dir: str):
        """Generates statistical reports and visualization."""
        
        # 1. Classification Report
        logger.info("Generating Classification Report...")
        report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
        print("\n" + "="*40)
        print("External Validation Classification Report")
        print("="*40)
        print(report)
        
        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Visualization (Seaborn style for publication quality)
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)
        
        # Calculate percentages
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.clip(min=1e-6) # Avoid division by zero
        
        # Annotations
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p * 100, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p * 100, c)
        
        heatmap = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
                              xticklabels=classes, yticklabels=classes)
        
        plt.title('External Validation Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300)
        logger.info(f"Confusion matrix saved to {save_path}")
        # plt.show() # Optional

def main(args):
    # 1. Load Data
    image_paths, df_metadata = DataIngestion.load_cohort(args.test_dir)
    
    if df_metadata.empty:
        logger.error("No images found in the dataset directory.")
        return

    # 2. Initialize Model
    inferencer = ModelInference(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint
    )
    
    # 3. Run Inference
    prompts = InferenceConfig.PROMPTS
    logits = inferencer.run_batch_inference(image_paths, prompts)
    
    # 4. Process Results
    pred_indices = np.argmax(logits, axis=1)
    
    pred_labels = [InferenceConfig.CLASS_NAMES[idx] for idx in pred_indices]
    
    df_metadata['predicted_label'] = pred_labels
    
    logger.info("Sample predictions:")
    for i in range(min(5, len(df_metadata))):
        img_name = df_metadata.index[i]
        pred = df_metadata.iloc[i]['predicted_label']
        conf = logits[i][pred_indices[i]]
        print(f"[{img_name}] -> {pred} (Conf: {conf:.4f})")

    # 5. Evaluate
    valid_mask = df_metadata['true_label'].isin(InferenceConfig.CLASS_NAMES)
    if not valid_mask.all():
        logger.warning(f"Found labels not in target classes. Filtering { (~valid_mask).sum() } samples...")
        df_filtered = df_metadata[valid_mask]
    else:
        df_filtered = df_metadata

    EvaluationEngine.generate_report(
        y_true=df_filtered['true_label'].tolist(),
        y_pred=df_filtered['predicted_label'].tolist(),
        classes=InferenceConfig.CLASS_NAMES,
        output_dir=os.path.dirname(args.checkpoint) # Save where model is
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run External Validation for Orbital-CLIP")
    
    parser.add_argument("--test-dir", type=str, default="dataset_eye_cut_ninth_easy2", 
                        help="Path to external validation dataset root")
    parser.add_argument("--model-name", type=str, default="ViT-B-32-256")
    parser.add_argument("--checkpoint", type=str, default="clip_model_checkpoints/clip_model_epoch_best.pt",
                        help="Path to fine-tuned model weights")
    
    args = parser.parse_args()
    
    main(args)
