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

- `backend`:`auto`(mlx→openai)/ `mlx` / `openai` / `apple`(需 proxy,見下)/ `groq`(雲端)
- `language` 用 whisper 碼(`zh`/`en`);apple 後端內部自動轉 BCP-47(`zh-Hans`)
- **反幻聽**:靜音/雜訊產生的段落(`no_speech_prob` 高、`compression_ratio > 2.4` 的迴圈)自動過濾
- mlx 模型已快取後,設 `HF_HUB_OFFLINE=1` 可跳過每次轉寫的 HuggingFace 網路檢查(省數百 ms + log 噪音)

### Groq 雲端後端

```python
segs = transcribe_audio("rec.mp4", backend="groq", language="zh")   # 需 GROQ_API_KEY
```

`GROQ_API_KEY` 取自環境變數或 `~/.env`。實測 whisper-large-v3 約 100× 實時,
嘈雜音源明顯比本地 medium 準;音訊會上傳,自行斟酌。實作細節:自動轉 64 kbps
mono mp3(25 MB 上限)、超長切 25 分鐘段並校正時間戳、繁中 prompt 偏置、
與本地路徑共用同一套幻聽過濾。

**失敗處理**(`groq_fallback=True`,預設):

| 情境 | 行為 |
|---|---|
| 每分鐘限流 429 | 依 `retry-after` 等待重試一次;成功就繼續用 Groq |
| 每日額度用盡 429 | 不重試,**自動改用本地後端**完成該次轉寫 |
| 5xx / 網路錯誤 | 重試一次,仍失敗則自動改用本地 |
| 401 金鑰無效、4xx 輸入錯誤 | **直接報錯**(fallback 只會掩蓋問題) |

`groq_fallback=False` 時上述可恢復錯誤改為拋出 `GroqUnavailable`。

`groq_usage_today()` 回傳當日額度狀況。**Groq 對 whisper 只回請求數的
rate-limit header**(`x-ratelimit-{limit,remaining,reset}-requests`,實測確認),
沒有 audio-seconds header,也沒有用量查詢 API,因此:

- `requests_remaining` / `requests_limit`:**Groq 回報的權威值**(上次請求時的快照)
- `audio_seconds`:本機累計的已送音訊秒數(其他機器共用同一金鑰時看不到)

文件宣稱免費層每日 28800 音訊秒上限,但**實測不強制**(2026-07-08 單日送出
40k+ 秒仍正常服務),因此不從中推算「剩餘額度」;真正的權威訊號是 429 回應。

輔助函式:`verify_groq_key(key)`(即時驗證)、`read_env_value` / `write_env_value`
(管理 `~/.env`,寫入時 chmod 600)。

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
