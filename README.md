# 基米看得见视频 · astrbot_plugin_video_vision

让 AstrBot 把视频作为**原生模态**直接交给 Gemini 理解，而不是抽取关键帧。

> AstrBot 核心目前只把视频写成一行文本占位符
> （`[Video Attachment: name X, path Y]`）交给模型，**模型看不到任何画面**。
> 本插件在 `on_llm_request` 阶段把视频转成内容块注入请求，
> 使模型获得完整的画面、动作、字幕与时间轴理解能力。

---

## 特性

- **原生视频理解**，非抽帧 —— 模型真的"看"视频
- **自动压缩**：超过体积闸门时用 ffmpeg 压缩，画质随画面复杂度自适应
- **体积上限保险丝**：即使画质档位拉满也不会压出超预算的文件
- **压缩缓存**：按源文件内容哈希缓存，重试/重复发送不重复压缩
- **全链路失败降级**：视频处理出错**绝不影响**正常文字对话
- **不污染对话历史**：注入的视频块标记为非持久化，不会在后续轮次被重发
- **`/videotest` 诊断指令**：逐步查看处理耗时与体积，排障一目了然
- **与「我会永远陪着你」陪伴体系兼容**（见下方兼容性说明）

---

## 安装

1. 把本插件目录放进 AstrBot 的 `data/plugins/`
2. 重启 AstrBot 或在 WebUI 插件页重载
3. 在 WebUI 插件配置里按需调整参数
4. 确认目标模型**勾选了「音频」模态**（见下方"为什么需要 audio 模态"）

### 依赖

- **ffmpeg / ffprobe**：压缩功能需要。请在系统 PATH 中提供，或在插件配置里填 `ffmpeg_path`。
  - 没有 ffmpeg 时插件仍可工作，但超限视频会被跳过而不是压缩。
- 无第三方 Python 依赖。

---

## 配置项

| 键 | 默认 | 说明 |
|---|---|---|
| `enable` | `true` | 总开关 |
| `require_gemini_provider` | `true` | 仅在 Gemini 系 provider 上注入。**建议保持开启** |
| `max_size_mb` | `13.0` | 体积闸门，超过则按 `oversize_action` 处理 |
| `compress_max_mb` | `8.0` | 压缩产物体积**上限**（保险丝），必须小于闸门 |
| `compress_crf` | `30` | **主控压缩程度**：越小画质越好、体积越大 |
| `compress_preset` | `veryfast` | x264 速度预设 |
| `max_duration_sec` | `60.0` | 最长时长，超出则截断。`0` 表示不限 |
| `oversize_action` | `compress` | `compress` / `reject` / `truncate` |
| `max_video_count` | `2` | 单条消息最多处理几个视频 |
| `handle_quoted_video` | `true` | 是否处理引用消息里的视频 |
| `treat_file_as_video` | `true` | **把视频文件附件也当作视频处理**（见下方「QQ 把视频当文件发」） |
| `hide_local_path` | `true` | **注入后抹除占位文本中的本机路径**（见下方「别把路径喂给模型」） |
| `notify_on_failure` | `true` | 失败时追加一句简短说明给模型 |
| `max_compress_seconds` | `45.0` | 压缩耗时上限，超时则降级 |
| `ffmpeg_path` | `""` | 留空则从 PATH 查找；也可填所在目录 |
| `cache_enabled` | `true` | 启用压缩缓存 |
| `cache_max_mb` | `512` | 缓存上限，超出按 LRU 淘汰 |
| `debug_log` | `false` | 详细调试日志 |

### 两个体积旋钮怎么配合

- **`compress_crf` 决定实际体积**，且随画面复杂度自适应：
  静态画面压得极小、上传更快；高熵画面自动保留更多细节。
- **`compress_max_mb` 只是上限**。实测：CRF 18 想产出 5.23MB，
  上限设 1MB 时会被 `maxrate` 压到 0.83MB。

---

## 体积与耗时的实测参考

经中转站上传的实测数据（**瓶颈是上行带宽，不是 token**）：

| 原始体积 | 处理耗时 | 说明 |
|---|---|---|
| 3.93 MB | ~38 s | |
| 7.88 MB | ~42 s | |
| 13.76 MB | ~60 s | 默认闸门附近 |
| 19.61 MB | ~105 s | 贴边 |
| 27.47 MB | ~183 s | 会超时 |

约 **7–9 秒 / MB**。默认闸门 13MB 是为 120s 超时留的余量；
若你把 provider `timeout` 调到 180s，可以把闸门相应放宽。

**Token 消耗很低**：约 72–108 token / 秒视频
（14 秒视频 ≈ 1000 token，跟一张 720p 图片差不多）。

---

## QQ 把视频当文件发（实测发现，重要）

**QQ / NapCat 会把用户发的视频作为「文件」而不是「视频」投递。** 实测日志：

```
04:19:58 有所思: [ComponentType.File]          ← 不是 ComponentType.Video
04:21:00 C:\...\data\temp\fileseg_<hash>.mp4   ← 6.16 MB
```

此时 AstrBot 核心走的是 `isinstance(comp, File)` 分支，产出的是

```
[File Attachment: name fileseg_xxx.mp4, path C:\...\fileseg_xxx.mp4]
```

而不是 `[Video Attachment: ...]`。

**如果插件只认 `[Video Attachment: ...]`，在这种环境下会永远不触发** —— 这正是本插件
早期版本踩过的坑。因此本插件同时解析两种占位符：

- `[Video Attachment: ...]`（来自 `Video` 组件）→ 无条件处理
- `[File Attachment: ...]`（来自 `File` 组件）→ 仅当**扩展名属于常见视频格式**时处理

普通文件（`.zip` / `.pdf` / `.mp3` 等）不会被误处理，也不会产生任何提示。

关闭 `treat_file_as_video` 可禁用第二条路径。

---

## 别把路径喂给模型（实测发现）

核心的占位文本**包含本机绝对路径**：

```
[File Attachment: name fileseg_xxx.mp4, path C:\path\to\fileseg_xxx.mp4]
```

模型是 agentic 的 —— 看到路径就会忍不住自己动手。实测日志：

```
05:04:17  [VideoVision] 视频已注入 ... 6.16MB     ← 视频已经送到模型了
05:05:45  使用工具：astrbot_execute_python {'code': 'import cv2...'}
05:05:46  Result: FPS: 30.0, Frame count: 87
```

**模型明明已经拿到视频，却还是去跑 cv2 抽帧、grep 知识库、web search** ——
一次回复拖了十几分钟，用户看到的就是"截完帧查知识库然后开始 web search"。

开启 `hide_local_path`（默认）后，**成功注入**的视频占位文本会被改写成：

```
[Video: fileseg_xxx.mp4]
```

模型仍然知道"有个视频、叫什么名字"，但不再被路径诱导去自己解析。
**注入失败时占位文本保持原样**（那时路径反而有用）。

---

## 为什么需要 audio 模态

本插件借道 `audio_url` 内容块传输视频。这不是巧合，而是因为
Gemini provider 的 `process_audio_url` 是 **MIME 无关**的：

```python
mime_type = url.split(":")[1].split(";")[0]   # data:video/mp4;base64,... -> video/mp4
return types.Part.from_bytes(data=..., mime_type=mime_type)
```

它直接信任 data URI 里的 MIME 类型，因此 `data:video/mp4` 会被构造成**视频 Part**。

**代价**：AstrBot 的模态清洗会把"模型未声明的 audio 块"替换成 `[Audio]` 文本。
所以目标模型必须勾选**「音频」**模态，否则视频会被清洗掉。
插件检测到这种情况会打 WARNING 并跳过注入，而不是静默发错内容。

---

## `/videotest` 诊断指令

仅管理员可用。

```
/videotest <视频文件绝对路径>
```

或回复一条含视频的消息，直接发送 `/videotest`。

输出示例：

```
🔍 视频链路诊断
文件: l10_probe.mp4
体积: 19.61 MB
时长: 10.0 s

ffmpeg : ✅ C:\path\to\ffmpeg.exe
ffprobe: ✅ C:\path\to\ffprobe.exe
配置   : 闸门=13.0MB 上限=8.0MB CRF=30 策略=压缩
判定   : 体积 19.6MB > 闸门 13.0MB → 压缩

✅ 准备成功（0.62s）
动作: 已压缩
产物: 1.35 MB / 10.0 s
编码: data URI 1.80 MB

预计上传耗时: 约 11s（provider 预算 180s）
✅ 在预算内
```

---

## 兼容性说明（「我会永远陪着你」）

本插件经过针对 `astrbot_plugin_private_companion` 的专项审计，遵守以下约束：

| 约束 | 原因 |
|---|---|
| **不碰** `image_urls` / `audio_urls` | 陪伴插件主动管理这两个字段，核心还会据此切换 provider |
| **不碰** `prompt` / `system_prompt` | 陪伴插件大量读写 |
| **不替换** `extra_user_content_parts` 列表 | 只就地 `append`，且每次重新读取属性（陪伴插件会替换整个列表） |
| **不删除** `[Video Attachment: ...]` 占位文本 | 陪伴插件与核心都不删它，保留最安全 |
| **不调用** `event.stop_event()` | 会中断所有更低优先级的钩子（含陪伴插件整条富化链） |
| **注入块标记** `_no_save` | 否则多兆 base64 会写进对话历史并在每轮重发 |
| **用** `AudioURLPart` 对象而非裸 dict | provider 侧 `assemble_context` 对裸 dict 会 `raise ValueError` |
| **钩子优先级** `-300000` | 比陪伴插件最低的 `-260000` 更晚，确保最后写入 |

同时，本插件与 `astrbot_plugin_together_companion`（「我会和你在一起」）
**互补不冲突**：后者用**抽帧**实现"一起看视频"的网页播放器场景，
本插件用**原生视频**处理"发我一段视频"的场景，触发路径不同。

---

## 工作原理

```
QQ 收到视频消息
   ↓
核心 astr_main_agent 写入占位符 TextPart（含本地路径）
   ↓
本插件 on_llm_request 钩子
   ├─ 扫描 extra_user_content_parts 里的视频占位符
   ├─ 探测体积/时长 → 超限则 ffmpeg 压缩（带缓存）
   ├─ 读文件 → base64 → data:video/mp4 URI
   └─ 追加 AudioURLPart(url=<data uri>).mark_as_temp()
   ↓
ProviderRequest.assemble_context()      ← 真实主链路
   ↓
provider.text_chat(contexts=[...], prompt=None)
   ↓
_prepare_conversation → process_audio_url → Part.from_bytes(mime="video/mp4")
   ↓
Gemini 原生视频理解
```

### 为什么不用 provider 适配器

主对话轮次的上下文由 `ProviderRequest.assemble_context()` 组装，runner 传
`contexts=` 且 **`prompt=None`**；而 `ProviderGoogleGenAI.text_chat` 只在
`prompt is not None` 时才调用 provider 侧的 `assemble_context`。

**因此覆写 provider 的 `assemble_context` 在主对话轮次上永远不会被调用** ——
只有图片描述、TTS 后处理等辅助调用会走到那里。
本插件因此选择 `on_llm_request` 钩子，**零配置改动**即可生效。

---

## 目录结构

```
astrbot_plugin_video_vision/
├── main.py                 插件入口：注册钩子与 /videotest
├── _conf_schema.json       WebUI 配置 schema
├── metadata.yaml
├── core/                   与 AstrBot 解耦的纯逻辑（便于日后上移核心）
│   ├── config.py           配置模型与容错
│   ├── media.py            ffmpeg/ffprobe 封装、压缩、缓存
│   ├── injector.py         体积闸门决策与压缩编排
│   └── placeholder.py      占位文本解析
└── hooks/
    └── llm_request.py      on_llm_request 注入逻辑
```

`core/` 下**不允许**出现 `import astrbot`，以便日后把这套逻辑搬进 AstrBot 核心。

---

## 已知限制

- 视频以 **inline base64** 传输，因此体积受 provider 超时预算约束；
  中转站若提供 File API 可显著放宽（当前实测的中转站没有）。
- 仅支持 **Gemini 系** provider（依赖其按 MIME 构造媒体 Part 的行为）。
- 模型必须声明 **audio** 模态（见上文）。
- 不支持多视频并行压缩；单条消息的视频串行处理。

## 许可

MIT
