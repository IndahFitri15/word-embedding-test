from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm

# Mengunduh tokenizer dari Hugging Face
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7")

# Mengunduh dan memuat model AutoModelForCausalLM dalam mode 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "BioMistral/BioMistral-7",
    load_in_8bit=True,
    device_map="auto"
)

# Jika Anda menggunakan PEFT model, pastikan untuk mengimpornya
from peft import PeftModel  # Pastikan modul ini diinstal

# Menggunakan model PEFT yang telah dilatih sebelumnya
# model = PeftModel.from_pretrained(model, "Indah1/FinalBioMistralTes10")
# model = PeftModel.from_pretrained(model, "Indah1/BioMistralTesss15")
model = PeftModel.from_pretrained(model, "Indah1/BioChat10")

# Memuat dataset
test = load_dataset("Indah1/ChatDoctor-split", split="test")

# Mengambil 10 entri pertama dari dataset test
test_sample = test.select(range(250))

# Menggunakan kolom 'input' untuk tokenisasi
encodings = tokenizer("\n\n".join(test_sample["input"]), return_tensors="pt")

max_length = model.config.max_position_embeddings
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Menentukan perangkat

# Memindahkan input_ids ke GPU
input_ids = encodings.input_ids.to(device)

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids[:, begin_loc:end_loc], labels=target_ids[:, begin_loc:end_loc])

        # Menghitung neg_log_likelihood
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

# Menghitung perplexity
ppl = torch.exp(torch.stack(nlls).mean())
print(f'Perplexity: {ppl.item()}')
