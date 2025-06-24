from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

model_path = "/mnt/f/ubuntu/deployment/model/Qwen2___5-1___5B-Instruct-GPTQ-Int4"

tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,local_files_only=True) # 强制在本地加载tokenizer

# 加载模型
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True, # 必须，允许加载qwen的自定义代码
    local_files_only=True, # 强制在本地加载
    quantize_config='gptq', # 必须，量化配置
    # 内存不足时
    # use_safetensors=True, # 必须，使用safetensors优化模型推理/训练速度
    # 量化模型
    # quantize_config = 'gptq' 
    # 对低显存GPU必要 优化 Transformer 模型推理/训练速度的参数 替换原始注意力计算为更高效的融合内核
    # inject_fused_attention = False  # 启用融合注意力   或者use_fused_attention
).to('cuda:0')


from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
app = FastAPI()

class Input(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(input: Input):
    prompt = input.prompt

    inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    outputs = model.generate(**inputs,max_new_tokens=200)

    return {"response":tokenizer.decode(outputs[0], skip_special_tokens=True)}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)