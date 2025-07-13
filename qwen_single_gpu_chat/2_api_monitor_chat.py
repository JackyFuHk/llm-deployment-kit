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

####### 监控指标
from prometheus_client import make_asgi_app, Gauge, Counter, Histogram
import psutil
metrics_app = make_asgi_app()  # Prometheus ASGI 中间件
# 定义自定义指标
GPU_MEMORY = Gauge("gpu_memory_usage", "GPU memory usage in MB", ["device"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")
TOKENS_GENERATED = Counter("tokens_generated_total", "Total tokens generated")  





####### 接口模块
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
import torch
app = FastAPI()

app.mount("/metrics/", metrics_app)  # 暴露指标端点

class Input(BaseModel):
    prompt: str

@app.post("/chat/")
async def chat(input: Input):
    # 记录请求开始时间（用于计算延迟）
    start_time = time.time()

    prompt = input.prompt

    inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    outputs = model.generate(**inputs,max_new_tokens=200)
    tokens_count = len(outputs[0])

    # 更新指标
    REQUEST_LATENCY.observe(time.time() - start_time)
    TOKENS_GENERATED.inc(tokens_count)

    # 记录显存占用
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            GPU_MEMORY.labels(device=f"cuda:{i}").set(torch.cuda.memory_allocated(i) / 1024 / 1024)

    return {"response":tokenizer.decode(outputs[0], skip_special_tokens=True)}


# # 后台线程定期更新系统指标
# def update_system_metrics():
#     while True:
#         # 添加 CPU/内存监控（可选）
#         # ...
#         time.sleep(10)

# Thread(target=update_system_metrics, daemon=True).start()
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)