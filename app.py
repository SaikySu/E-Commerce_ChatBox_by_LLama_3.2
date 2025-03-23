import streamlit as st
st.set_page_config(layout="wide") 

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

MODELS = {
    "LLaMA 3": "llama3_dpo_final",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load mô hình và tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_name):
    model_path = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    if "llama" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        ).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        ).to(device)

    return tokenizer, model

# Tải mô hình 
tokenizer_llama, model_llama = load_model_and_tokenizer("LLaMA 3")

# Hàm sinh phản hồi
def generate_response(prompt, tokenizer, model, model_type):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if "llama" in model_type.lower():
        output = model.generate(
            **inputs, max_length=512, do_sample=True, top_p=0.9, temperature=0.7
        )
    else:
        output = model.generate(
            **inputs, max_length=512, do_sample=True, top_p=0.9, temperature=0.7,
            decoder_start_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Giao diện người dùng
st.title("Chatbot Thương mại điện tử: LLaMA 3")

user_input = st.text_input("Nhập câu hỏi:")

if user_input:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔷 LLaMA 3")
        with st.spinner("Đang sinh phản hồi từ LLaMA 3..."):
            response_llama = generate_response(user_input, tokenizer_llama, model_llama, "llama")
        st.markdown(f"**Phản hồi:**\n\n{response_llama}")
