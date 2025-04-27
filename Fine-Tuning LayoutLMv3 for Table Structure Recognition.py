# Step: Download and Extract PubTabNet Dataset inside Colab

!wget -O PubTabNet.tar.gz https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz
!mkdir -p PubTabNet_2.0.0
!tar -xvzf PubTabNet.tar.gz -C PubTabNet_2.0.0
!ls PubTabNet_2.0.0


# ✅ Step 0: Install Dependencies
#!pip install transformers datasets pytesseract
#!apt-get install -y tesseract-ocr

# ✅ Step 1: Imports
import os, json, torch, shutil
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)
from google.colab import files

# ✅ Step 2: Define Paths
BASE_IMG_DIR = "/content/PubTabNet_2.0.0/pubtabnet"
JSONL_FILE   = "/content/PubTabNet_2.0.0/pubtabnet/PubTabNet_2.0.0.jsonl"

# ✅ Step 3: Dataset Class using 'cells' and processor for input_ids
class PubTabNetJSONLDataset(Dataset):
    def __init__(self, jsonl_file, img_base_dir, split="train", processor=None, max_samples=500):
        self.samples = []
        self.img_base_dir = img_base_dir
        self.processor = processor
        self.max_samples = max_samples

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("split") != split:
                    continue
                image_path = os.path.join(img_base_dir, split, entry["filename"])
                cells = entry["html"].get("cells", [])
                if os.path.exists(image_path) and cells:
                    self.samples.append({"image_path": image_path, "cells": cells})
                    if len(self.samples) >= max_samples:
                        break
        print(f"[INFO] Loaded {len(self.samples)} samples for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        words, boxes, labels = [], [], []

        for cell in sample["cells"]:
            tokens = cell.get("tokens", [])
            if not tokens or not "".join(tokens).strip():
                continue
            text = " ".join(tokens)
            words.append(text)
            boxes.append(cell.get("bbox", [0, 0, 224, 224]))
            labels.append(1 if "<b>" in tokens[0] else 0)  # dummy: header = 1 if <b>

        if not words:
            words = ["empty"]
            boxes = [[0, 0, 224, 224]]
            labels = [0]

        encoding = self.processor(
            text=words,
            boxes=boxes,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        label_padded = labels + [-100] * (encoding["input_ids"].size(1) - len(labels))
        label_padded = label_padded[:512]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": torch.tensor(label_padded, dtype=torch.long),
        }

# ✅ Step 4: Load Processor (disable OCR) and Datasets
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

train_dataset = PubTabNetJSONLDataset(
    jsonl_file=JSONL_FILE,
    img_base_dir=BASE_IMG_DIR,
    split="train",
    processor=processor,
    max_samples=200
)

val_dataset = PubTabNetJSONLDataset(
    jsonl_file=JSONL_FILE,
    img_base_dir=BASE_IMG_DIR,
    split="val",
    processor=processor,
    max_samples=100
)

# ✅ Step 5: Collate Function
def collate_fn(batch):
    return {
        "input_ids":      torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "bbox":           torch.stack([item["bbox"] for item in batch]),
        "pixel_values":   torch.stack([item["pixel_values"] for item in batch]),
        "labels":         torch.stack([item["labels"] for item in batch]),
    }

# ✅ Step 6: Load Model and Define Training Arguments
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=2
).to("cpu")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,  # Increased batch size
    per_device_eval_batch_size=4,
    num_train_epochs=1,             # Only 1 epoch
    eval_strategy="epoch",
    learning_rate=5e-5,
    logging_steps=50,
    save_steps=200,
    report_to="none",
    no_cuda=True,                   # Keep True if no GPU
    gradient_accumulation_steps=1,   # Accumulate less
)


# ✅ Step 7: Trainer and Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

trainer.train()

# ✅ Step 8: Save and Download Model
model.save_pretrained("FineTuned_PubTabNet")
shutil.make_archive("FineTuned_PubTabNet", 'zip', "FineTuned_PubTabNet")
files.download("FineTuned_PubTabNet.zip")




# ✅ Step 9: Testing the Fine- Tuned Model
# Imports
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import pytesseract
from google.colab import files

# Upload your table image (e.g. simple_table.png or any .png/.jpg)
print("🔽 Please upload a table image to test:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
print(f"Loaded image: {image_path}")

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")

# Run OCR to get words + bounding boxes
ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
words, boxes = [], []
for i, word in enumerate(ocr_data["text"]):
    if not word.strip():
        continue
    x, y, w, h = (ocr_data[k][i] for k in ("left","top","width","height"))
    words.append(word)
    boxes.append([x, y, x + w, y + h])

print(f"Detected {len(words)} OCR tokens.")

# Initialize your processor & fine-tuned model
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "/content/FineTuned_PubTabNet"  # adjust if your folder is elsewhere
).to("cpu")
model.eval()

# Prepare inputs for the model
encoding = processor(
    text=words,
    boxes=boxes,
    images=image,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512,
)

# move to CPU
for k,v in encoding.items():
    encoding[k] = v.to("cpu")

# Forward pass & gather predictions
with torch.no_grad():
    outputs = model(**encoding)
logits = outputs.logits  # (1, seq_len, num_labels)
pred_ids = logits.argmax(-1).squeeze().tolist()  # (seq_len,)

# filter out padding
mask = encoding["attention_mask"].squeeze().tolist()
predicted_labels = [lab for lab, m in zip(pred_ids, mask) if m == 1]

print("\n✅ Predicted labels for each token (0=cell, 1=header):")
print(predicted_labels)
