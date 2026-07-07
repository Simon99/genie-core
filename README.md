# genie-core

Genie 系列工具的共用基礎庫:語音轉文字、影片截圖、PDF 拆頁、LM Studio LLM client、報告產生。

## 安裝

```bash
pip install -e .            # 基本(只有 requests 依賴)
pip install -e ".[pdf]"     # + PyMuPDF(PDF 拆頁)
pip install -e ".[mlx]"     # + mlx-whisper(Apple Silicon 語音轉文字)
pip install -e ".[whisper]" # + openai-whisper(CPU fallback)
```

系統依賴:`ffmpeg`(音訊抽取/截圖必要,`brew install ffmpeg`)。

## 模組總覽

| 模組 | 功能 |
|---|---|
| `genie_core.audio.transcribe` | 三後端語音轉文字(mlx / openai-whisper / Apple Speech proxy) |
| `genie_core.audio.srt` / `loader` | SRT 讀寫、transcript 檔載入(.json/.srt) |
| `genie_core.video` | ffmpeg 場景偵測、截圖、字幕燒錄 |
| `genie_core.pdf.split` | PDF 逐頁轉圖檔(PyMuPDF) |
| `genie_core.llm` | LM Studio client、JSON 抽取、分層合併 |
| `genie_core.report` / `text` | HTML 報告產生(自動 escape)、時間格式化 |

## LLM client(LM Studio)

```python
from genie_core.llm import LMStudioClient

c = LMStudioClient()                      # model=None → 自動挑選
c = LMStudioClient(kind="vision")         # 挑有 vision 能力的模型
print(c.model)                            # 實際選中的模型
print(c.get_context_length())             # 模型 context 長度(如 262144)

text = c.complete("問題", system="你是...", max_tokens=4096)
desc = c.vision("描述這張圖", "/path/img.png", max_tokens=2000)
```

重點行為:

- **API**:優先走 LM Studio 原生 `POST /api/v1/chat`(reasoning 與 message 由 API 分離,影像用 `{type:"image", data_url}`);舊版 LM Studio 404 時自動 fallback 到 `/v1/chat/completions`。
- **模型自動挑選**:讀 `GET /api/v1/models` 的真實能力標記(`capabilities.vision`、`type`),**已載入的模型優先**;指定的 model id 不存在時警告並 fallback。優先序可用環境變數覆寫:
  ```bash
  export GENIE_TEXT_MODELS="qwen3.6,glm-4.7"     # 逗號分隔的子字串,依序匹配
  export GENIE_VISION_MODELS="qwen3-vl-30b"
  ```
- **thinking 模型防護**:content 空但有 reasoning 時,自動以加倍的 max_tokens 重試一次,仍空則報明確錯誤。**經驗:qwen3.5 系 / glm-4.7-flash 的推理會隨預算膨脹,建議直接用非 thinking 模型(如 qwen3.6-35b-a3b-turboquant)**。
- 一律傳 `max_tokens`:防 runaway 生成(曾實測 5 分鐘 read timeout)。

## 語音轉文字

```python
from genie_core.audio import transcribe_audio

segs = transcribe_audio("rec.mp4", backend="mlx", model="medium", language="zh",
                        initial_prompt="會議詞彙:ChromaDB、PVT")  # 熱詞偏置
# → [{"start": 0.0, "end": 2.5, "text": "..."}]
```

- `backend`:`auto`(mlx→openai)/ `mlx` / `openai` / `apple`(需 proxy,見下)
- `language` 用 whisper 碼(`zh`/`en`);apple 後端內部自動轉 BCP-47(`zh-Hans`)
- **反幻聽**:靜音/雜訊產生的段落(`no_speech_prob` 高、`compression_ratio > 2.4` 的迴圈)自動過濾
- mlx 模型已快取後,設 `HF_HUB_OFFLINE=1` 可跳過每次轉寫的 HuggingFace 網路檢查(省數百 ms + log 噪音)

## Apple Speech proxy(swift-cli/)

macOS Speech Framework 的 HTTP 代理,繞過 SSH 環境的 TCC 限制。**只綁 127.0.0.1**。

```bash
cd swift-cli && ./install-proxy.sh   # 編譯 + 安裝 LaunchAgent(com.genie.speech-proxy, port 5300)
curl http://127.0.0.1:5300/health
```

已知坑:

- **更新 binary 必須原子替換**(install script 已處理):直接 `cp` 覆蓋執行中的 binary 會被 kernel 以 OS_REASON_CODESIGNING 殺掉
- binary 更新後 TCC 語音辨識授權會失效,需在 GUI session 重新點一次授權
- 授權未給時 proxy 以 exit(0) 退出(配合 KeepAlive/SuccessfulExit 不會無限重啟彈窗)
