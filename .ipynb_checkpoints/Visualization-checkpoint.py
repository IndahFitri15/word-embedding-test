# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Inisialisasi TensorBoard
writer = SummaryWriter(log_dir="./runs/weat_visualization")

# Load model dan tokenizer
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
model = AutoModelForCausalLM.from_pretrained(
    "BioMistral/BioMistral-7B",
    load_in_8bit=True,
    device_map="auto",
    output_hidden_states=True  # Ensure hidden states are available
)
model = PeftModel.from_pretrained(model, "Indah1/BioChat15")

# Fungsi untuk mendapatkan embedding kata
def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.logits.mean(dim=1).squeeze().numpy()  # Gunakan logits
    return embedding

# List kata-kata untuk divisualisasikan
X = ['male', 'man', 'boy', 'he', 'father', 'brother', 'uncle', 'husband', 'son', 'gentleman', 'sir']
Y = ['female', 'woman', 'girl', 'she', 'mother', 'aunt', 'sister', 'wife', 'daughter', 'lady', 'madam']
A = ['heart_disease', 'lung_cancer', 'depression', 'obesity', 'liver_disease', 'pancreatic_cancer', 'colon_cancer', 'type_2_diabetes', 'hypertension']
B = ['autoimmune_diseases', 'breast_cancer', 'anxiety_disorders', 'osteoporosis', 'thyroid_disease', 'eating_disorders', 'migraine', 'anemia', 'menstruation']

# Gabungkan semua kata
all_words = X + Y + A + B
embeddings = np.array([get_embedding(word) for word in all_words])

# Gunakan PCA untuk mereduksi dimensi menjadi 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot embeddings
plt.figure(figsize=(14, 10))
colors = ['red'] * len(X) + ['blue'] * len(Y) + ['green'] * len(A) + ['purple'] * len(B)
for i, word in enumerate(all_words):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=colors[i])
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], word, fontsize=12)

plt.title("2D PCA Visualization of Word Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

# Simpan plot ke TensorBoard
writer.add_figure("Word Embeddings Visualization", plt.gcf())
writer.close()

# Tampilkan plot
plt.show()

