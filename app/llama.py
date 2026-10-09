import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

tokenizer = AutoTokenizer.from_pretrained("AiModel/llama")

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="AiModel/llama",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


class ChatRequest(BaseModel):
    prompt:str


# Api Function
@router.post("/chat-llama")
def chat(req:ChatRequest):
    
    # Process input 
    message = [{"role":"user","content":req.prompt}]

    # Input token 
    InputToken = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
	    tokenize=True,
	    return_dict=True,
	    return_tensors="pt",
    ).to(model.device)

    # Input To ->>>>>> Model ----------> Output Token
    OutputToken = model.generate(
        **InputToken,
        max_new_tokens=128
    )


	# we can finshed properly 

    # Output Token ->>>>>> Output Text 
    result = tokenizer.decode(OutputToken[0][InputToken['input_ids'].shape[-1]:],skip_special_tokens=True)

    return {"response":result}
    
