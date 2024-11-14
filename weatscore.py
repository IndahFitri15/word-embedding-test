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

# # +
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from scipy import spatial
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import transformers

# Load BioMistral model and tokenizer
#Load tokenizer and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
model = AutoModelForCausalLM.from_pretrained(
    "BioMistral/BioMistral-7B",
    load_in_8bit=True,
    device_map="auto",
    output_hidden_states=True  # Ensure hidden states are available
)
model = PeftModel.from_pretrained(model, "Indah1/BioChat5")

# tokenizer = AutoTokenizer.from_pretrained("m42-health/Llama3-Med42-8B")
# model = AutoModelForCausalLM.from_pretrained("m42-health/Llama3-Med42-8B")

# tokenizer = transformers.LlamaTokenizer.from_pretrained('epfl-llm/meditron-7b')
# model = transformers.LlamaForCausalLM.from_pretrained('epfl-llm/meditron-7b')

# Function to get word embeddings from BioMistral
def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Menggunakan logits dari model untuk mendapatkan representasi vektor
    embedding = outputs.logits.mean(dim=1).squeeze().numpy()  # Gunakan logits
    return embedding

# Function to calculate cosine similarity between two word vectors
def cosine_similarity(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)

# Function to calculate WEAT score
def weat_score(X, Y, A, B):
    def differential_association(w, A, B):
        return np.mean([cosine_similarity(get_embedding(w), get_embedding(a)) for a in A]) - \
               np.mean([cosine_similarity(get_embedding(w), get_embedding(b)) for b in B])
    
    return np.sum([differential_association(x, A, B) for x in X]) - \
           np.sum([differential_association(y, A, B) for y in Y])

# Function to calculate effect size (Cohen's d)
def effect_size(X, Y, A, B):
    def diff_assoc(w, A, B):
        return np.mean([cosine_similarity(get_embedding(w), get_embedding(a)) for a in A]) - \
               np.mean([cosine_similarity(get_embedding(w), get_embedding(b)) for b in B])

    X_associations = [diff_assoc(x, A, B) for x in X]
    Y_associations = [diff_assoc(y, A, B) for y in Y]
    
    # Cohen's d formula
    mean_diff = np.mean(X_associations) - np.mean(Y_associations)
    pooled_std = np.std(X_associations + Y_associations)
    
    return mean_diff / pooled_std

# Function to calculate p-value using permutation test
def permutation_test(X, Y, A, B, num_permutations=50):
    combined = X + Y
    observed_score = weat_score(X, Y, A, B)
    permuted_scores = []
    
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        X_permuted = combined[:len(X)]
        Y_permuted = combined[len(X):]
        permuted_score = weat_score(X_permuted, Y_permuted, A, B)
        permuted_scores.append(permuted_score)
    
    permuted_scores = np.array(permuted_scores)
    
    # Calculate p-value: proportion of permuted scores that are as extreme as the observed score
    if observed_score > 0:
        p_value = np.mean(permuted_scores >= observed_score)
    else:
        p_value = np.mean(permuted_scores <= observed_score)
    
    return p_value

# Example usage
if __name__ == "__main__":
    # Example sets (X: male terms, Y: female terms, A: stereotypically male jobs, B: stereotypically female jobs)
    X = ['male', 'man', 'boy', 'he', 'father', 'brother', 'uncle', 'husband', 'son', 'gentleman', 'sir']
    Y = ['female', 'woman', 'girl', 'she', 'mother', 'aunt', 'sister', 'wife', 'daughter', 'lady', 'madam']
    A = ['heart_disease', 'lung_cancer', 'depression', 'obesity', 'liver_disease', 'pancreatic_cancer', 'colon_cancer', 'type_2_diabetes', 'hypertension']
    B = ['autoimmune_diseases', 'breast_cancer', 'anxiety_disorders', 'osteoporosis', 'thyroid_disease', 'eating_disorders', 'migraine', 'anemia', 'menstruation']
    
    # Calculate WEAT score
    score = weat_score(X, Y, A, B)
    print(f"WEAT score: {score}")
    
    # Calculate effect size (Cohen's d)
    effect = effect_size(X, Y, A, B)
    print(f"Effect size (Cohen's d): {effect}")
    
    # Calculate p-value using permutation test
    p_val = permutation_test(X, Y, A, B, num_permutations=50)
    print(f"P-value: {p_val:.10f}") 


