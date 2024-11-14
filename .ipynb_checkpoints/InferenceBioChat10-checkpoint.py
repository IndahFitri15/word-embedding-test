import gradio as gr
import transformers
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Mengunduh tokenizer dari Hugging Face
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")

# Mengunduh dan memuat model AutoModelForCausalLM dalam mode 8-bit dan secara otomatis memetakan ke perangkat yang tersedia
model = AutoModelForCausalLM.from_pretrained(
    "BioMistral/BioMistral-7B",
    load_in_8bit=True,      # Menggunakan mode 8-bit untuk mengurangi penggunaan memori
    device_map="auto"       # Secara otomatis memetakan model ke perangkat yang tersedia (CPU/GPU)
)

# Menggunakan model PEFT yang telah dilatih sebelumnya
model = PeftModel.from_pretrained(model, "Indah1/BioChat10")


generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.15,
)

def generate_response(instruction):
    PROMPT = f"""Below is an instruction that describes a task. You are as a Doctor please give a recomendation treat!.

### Instruction:
{instruction}

### Response:"""

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].cuda()

    print("Generating...")
    generation_output = model.generate(
        input_ids=input_ids,
        # generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    for s in generation_output.sequences:
        result = tokenizer.decode(s).split("### Response:")[1]
    return result

# Create Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="BioChat for Consultation",
    description="Ask something about your symptomps."
)

# Launch the Gradio interface
iface.launch(share=True)
