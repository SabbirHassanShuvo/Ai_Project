
# pyrefly: ignore [missing-import]
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='openai/whisper-small',
    local_dir='./AiModel/whisper-small',
    token=True
)