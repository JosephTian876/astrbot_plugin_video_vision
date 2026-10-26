# 基米看得见视频 · astrbot_plugin_video_vision

让模型真正**看见视频**，而不是抽取关键帧。支持两种模式：

- **openai_compat（默认）**：插件把视频发给你配置的 **OpenAI 兼容视觉服务**做理解，结果以文字注入对话 —— 主对话模型**可以是任意模型**
- **gemini_native**：把视频作为**原生模态**内容块注入当前请求，由 Gemini 主 provider 亲眼看（旧路径，完全保留）

> AstrBot 核心目前只把视频写成一行文本占位符
> （`[Video Attachment: name X, path Y]`）交给模型，**模型看不到任何画面**。
> 本插件在 `on_llm_request` 阶段接管视频附件：要么转成内容块注入请求让
> Gemini 亲眼看（gemini_native），要么先发给自定义的 OpenAI 兼容视觉服务
> 做理解、再把结果以文字随请求带给主模型（openai_compat）。

---

## 特性

- **双模式**：`openai_compat` 自定义 OpenAI 兼容视觉服务商（base_url/key/model 全可配），`gemini_native` 原生注入 Gemini，随时切换
- **原生视频理解**，非抽帧 —— 模型真的"看"视频，画面、动作、字幕、时间轴都在
- **带问题理解**：用户发视频时附带的文字会一起发给视觉模型，得到针对性回答
- **自动压缩**：超过体积闸门时用 ffmpeg 压缩，画质随画面复杂度自适应
- **体积硬上限**：压缩用 `-fs` 按字节数硬截断，产物不会显著超过 `compress_max_mb`
  （极小预算时会牺牲**时长**，见下方「两个体积旋钮怎么配合」）
- **压缩缓存**：按源文件内容哈希缓存，重试/重复发送不重复压缩
- **全链路失败降级**：视频处理出错**绝不影响**正常文字对话
- **不污染对话历史**：注入的视频块标记为非持久化，不会在后续轮次被重发
- **`/videotest` 诊断指令**：逐步查看处理耗时、体积、provider 与三道闸门
- **与「我会永远陪着你」陪伴体系兼容**（见下方兼容性说明）

---

## 两种模式怎么选

| | openai_compat（默认） | gemini_native |
|---|---|---|
| 主对话 provider 要求 | 任意 | 必须是 Gemini 系 |
| 视频谁来看 | 配置里的视觉模型 | 主对话模型本人 |
| 主模型拿到 | 视觉模型的文字理解 | 原生视频画面 |
| 额外配置 | base_url / key / model | 无（要求 audio 模态） |

openai_compat 的主模型上下文是「文字转述」，适合主聊天模型看不了视频的场景；
gemini_native 的理解上限更高（画面细节、时间轴追问），但要求 provider 支持。

### OpenAI 兼容接口的视频格式坑（实测 2026-09-29，Gemini 系中转站）

视频在 OpenAI 兼容 chat/completions 上**只有这一种形态有效**：

```json
{"type": "image_url", "image_url": {"url": "data:video/mp4;base64,..."}}
```

名字叫 image_url，但 MIME 由 data URI 头部决定，端点会按视频处理（实测 prompt token 262 → 416，模型能正确描述画面）。而`video_url` / `file` /
`input_video` 三种「看起来更标准」的形态全部 **HTTP 200 但视频被静默丢弃**（prompt token 不变、模型报 NO_VIDEO）。本插件按实测结论实现。

Responses API（`/responses`）同样实测支持：视频用 `input_image` + data URI 有效（`input_file` 会被静默丢弃），插件按地址结尾自动构造对应协议的请求。

---

## 安装

1. 把本插件目录放进 AstrBot 的 `data/plugins/`
2. 重启 AstrBot 或在 WebUI 插件页重载
3. 在 WebUI 插件配置里选择模式并调整参数：
   - **openai_compat（默认）**：填 `openai_base_url` / `openai_api_key` / `openai_model`（模型须支持视频，如 gemini-3.x-flash 系）。主对话 provider 任意。
   - **gemini_native**：主对话 provider 必须是 Gemini 系，且目标模型**勾选了「音频」模态**（见下方"为什么需要 audio 模态"）

### 依赖

- **ffmpeg / ffprobe**：压缩功能需要。请在系统 PATH 中提供，或在插件配置里填 `ffmpeg_path`。
  - 没有 ffmpeg 时插件仍可工作，但超限视频会被跳过而不是压缩。
- 无第三方 Python 依赖（`openai_compat` 复用 AstrBot 自带的 `httpx`，它随
  `google-genai` 一起安装；若删掉那条依赖会走 ImportError 兜底并给出提示）。
- AstrBot **>= 4.24.1**（`mark_as_temp` 自 4.24.1 起；占位符与 `modalities`
  清洗自 4.23.5 起）。

---

## 配置项

| 键 | 默认 | 说明 |
|---|---|---|
| `enable` | `true` | 总开关 |
| `provider_mode` | `openai_compat` | `openai_compat`（默认）/ `gemini_native`。视觉服务三项没配齐时**自动降级为 `gemini_native`**，因此老配置直接升级不会失效 |
| `openai_base_url` | `""` | 视觉服务**完整接口地址**，写全到 `/chat/completions` 或 `/responses`（按结尾自动选协议）；只填到 `/v1` 兜底前者 |
| `openai_api_key` | `""` | 视觉服务密钥 |
| `openai_model` | `gemini-3.8-flash-high` | 视觉模型，须支持视频 |
| `openai_timeout_sec` | `180` | 视觉服务超时（上传约 7-9s/MB）。既是单阶段超时，也是单个视频的整体硬上限 |
| `openai_max_tokens` | `2048` | 视觉回答 token 上限 |
| `vision_with_question` | `true` | 把用户随视频附带的文字一起发给视觉模型 |
| `vision_describe_prompt` | 见 schema | 无附带文字时的通用描述提示词 |
| `vision_qa_prompt` | 见 schema | 带问题提示词模板，`{question}` 会被替换 |
| `vision_verify_ssl` | `true` | 校验视觉服务证书；中转站证书有问题时可关（自担风险） |
| `require_gemini_provider` | `true` | **仅 gemini_native 模式生效**，建议保持开启 |
| `additional_provider_types` | `[]` | **仅 gemini_native 模式生效**：在默认 Gemini 系白名单外额外放行的 provider type（如走 OpenAI 协议的 Gemini 代理）。填错可能被上游 422 拒收 |
| `max_size_mb` | `13.0` | 体积闸门，超过则按 `oversize_action` 处理（上限 2000） |
| `compress_max_mb` | `8.0` | 压缩产物体积**硬上限**（传给 `-fs`），必须小于闸门；极小预算时产物会提前封口、**时长变短** |
| `compress_crf` | `30` | **主控压缩程度**：越小画质越好、体积越大 |
| `compress_preset` | `veryfast` | x264 速度预设 |
| `max_duration_sec` | `60.0` | 最长时长，超出则截断。`0` 表示不限（此时 `oversize_action=truncate` 会退化成压缩，启动会告警） |
| `oversize_action` | `compress` | `compress` / `reject` / `truncate`。`truncate` 按**体积预算**估算截断时长（见下） |
| `max_video_count` | `2` | 单条消息最多处理几个视频 |
| `handle_quoted_video` | `true` | 是否处理引用消息里的视频 |
| `treat_file_as_video` | `true` | **把视频文件附件也当作视频处理**（见下方「QQ 把视频当文件发」） |
| `hide_local_path` | `true` | **注入后抹除占位文本中的本机路径**（见下方「别把路径喂给模型」） |
| `notify_on_failure` | `true` | 失败时追加一句简短说明给模型 |
| `max_compress_seconds` | `45.0` | 压缩耗时**墙钟**上限（超时终止整棵进程树），超时则降级 |
| `ffmpeg_path` | `""` | 留空则从 PATH 查找；也可填所在目录 |
| `cache_enabled` | `true` | 启用压缩缓存 |
| `cache_max_mb` | `512` | 缓存上限，超出按 LRU 淘汰（下限 16；**单个产物大于此值时不进缓存**） |
| `debug_log` | `false` | 详细调试日志 |

### 两个体积旋钮怎么配合

- **`compress_crf` 决定实际体积**，且随画面复杂度自适应：
  静态画面压得极小、上传更快；高熵画面自动保留更多细节。
- **`compress_max_mb` 是硬上限**，通过 ffmpeg 的 `-fs <字节数>` 实现：写满即
  停止复用。实测把上限设 0.5MB 得到 532KB（溢出不到一个 GOP），而不再像 0.2.0
  那样只靠 `-maxrate`/`-bufsize`（实测会超 1.3–3.2 倍）。
  代价：预算极小时 ffmpeg 会**提前封口**，产物时长比预期短 —— 此时
  `/videotest` 的「说明」里会写明「受 -fs 体积硬上限约束，产物比预期更短」。
- **`oversize_action=truncate` 现在真的按体积截断**：估算
  `truncate_to ≈ duration × max_size_mb ÷ size_mb`（留 10% 余量、最小 1 秒），
  再与 `max_duration_sec` 取较小值。所以 19MB/10s 的视频配 13MB 闸门，
  会被截到约 6 秒，而不是原样整段上传。

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

开启 `hide_local_path`（默认）后，**成功注入**的视频占位文本会被改写成
（按模式二选一，都是整句中文/英文说明，不再包含任何路径）：

```
[视频内容已随本消息直接提供（文件名 fileseg_xxx.mp4）。你已经在本次请求中看到了
这段视频，无需再用任何工具去查找或读取该文件。]
```

```
[视频 fileseg_xxx.mp4 已交给视觉模型分析，分析结果已随本消息以文字形式提供。
你无需、也无法读取该视频文件，不要尝试用代码工具去解析它。]
```

模型仍然知道"有个视频、叫什么名字"，但不再被路径诱导去自己解析。
**注入失败时占位文本保持原样**（那时路径反而有用）。

> 改写的是**文本内容本身**，不是"删除占位符"。如果下游插件依赖
> `[Video Attachment` 这个前缀做判断，请关闭 `hide_local_path`。

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

仅管理员可用。三种入口，任选一种：

```
/videotest <视频文件绝对路径>
```

1. **引用（回复）那条视频消息**，再发送 `/videotest`。被引用消息的消息段在
   `Reply.chain` 里，插件就是从这里取视频的（`Reply.message_str` 只是文本，
   视频消息的文本通常是空的）。
2. **视频和 `/videotest` 写在同一条消息里**（平台支持图文混排时）。
3. 上面都不方便时给绝对路径。

> ⚠️ **"先单独发一个视频、再单独发 `/videotest`" 是不行的**。那是两条独立消息、
> 两个独立事件：处理 `/videotest` 时，上一条消息早已处理完，
> `event.message_obj.message` 里只有后一条，插件没有任何途径回头看它。
> 真机证据（2026-10-26 11:34）：11:34:35 发视频、11:34:40 发 `/videotest` → 只拿到
> 用法提示；11:34:52 改成**引用**那条视频再发 `/videotest` → 正常出诊断。

找不到视频时，回复会**说清是哪一种情况**（本条消息和引用里都没有视频 / 引用的
消息里没有视频 / 平台没把被引用消息的内容交给插件），不会只把用法再贴一遍。
最后一种（`Reply.chain` 为空）还会按 OneBot 的 `get_msg` 重取一次；仍取不到就
用绝对路径那条路。

openai_compat 模式下会**真实调用一次视觉服务**并回显回复节选，可端到端验证 base_url / key / 模型 / 网络 / 证书是否可用。

输出示例（gemini_native，真跑 21.79MB / 20s 的测试片）：

```
🔍 视频链路诊断
文件: demo.mp4
体积: 21.79 MB
时长: 20.0 s

ffmpeg : ✅ D:\ffmpeg\bin\ffmpeg.exe
ffprobe: ✅ D:\ffmpeg\bin\ffprobe.exe
配置   : 闸门=13.0MB 上限=8.0MB CRF=30 策略=压缩
编码上限: data URI 18MB ≈ 原始 13.5MB（编码后超过必失败，与体积闸门无关）
模式   : gemini_native
凭据   : 视觉服务 ❌ 缺 openai_base_url、openai_api_key
provider: type=googlegenai_chat_completion id=gemini-2.5-flash modalities=[text, image, audio]
闸门   : ⓪ File 组件扩展名筛选 → 本指令用显式路径，不适用
         ① 类型 googlegenai_chat_completion ✅ 在允许列表
         ② modalities 含 audio ✅
         ③ 视觉服务凭据 ❌ 缺 openai_base_url、openai_api_key（本模式不使用）
判定   : 体积 21.8MB > 闸门 13.0MB → 压缩
prompt : （gemini_native 不调用视觉服务，无 prompt）

✅ 准备成功（0.64s）
动作: 已压缩
产物: 1.79 MB / 20.0 s
编码: data URI 2.38 MB
MIME   : video/mp4（压缩产物，固定为 mp4）
说明: 体积 21.8MB > 闸门 13.0MB -> 1.8MB

预计上传耗时: 约 14s（provider 预算 180s）
✅ 在预算内
```

几处读法：

- **凭据**只在 openai_compat 模式有意义；gemini_native 下显示为缺失是正常的
  （第 ③ 行也标了"本模式不使用"）。
- **provider / 闸门** 三行是 0.2.1 新增的：`provider type`、`modalities`、以及
  ⓪①②③ 四道判定的结果，与钩子里的判定代码完全一致。
- **prompt** 是 openai_compat 模式下真正会发给视觉服务的提示词（压成一行、
  截断 200 字符）。若 `vision_qa_prompt` 里没有 `{question}`，这里会提示。
- **action** 现在只由"是否真的传了 `-t`"决定：`策略=压缩` 但 `max_duration_sec`
  真的触发了截断时，动作会显示"已截断"而不是"已压缩"。
- **编码上限**是**第二道**体积限制：data URI 编码后 18MB（≈ 原始 13.5MB），
  与 `max_size_mb` 无关。`判定` 行已经把它算进来 —— 闸门调到 13.5MB 以上时，
  原本的"未超限，直接送"会变成提前说明"编码后约 XMB > 编码上限 18MB，实际会以
  `too_large` 失败"；压缩目标（`compress_max_mb`）超过同一阈值时也会在判定行提示。
- **时长读不出来**（`.mp4` 等视频容器 + 有 ffprobe 却解析失败）时，判定行直接写
  「无法解析（时长读取失败）→ 会以 corrupt 失败」，不再先说"未超限，直接送"、
  下一段再报 `分类: corrupt`。
- **扩展名不在已知视频容器内**时（如 `.gif`），判定行上方会多一行 `提醒   :`
  说明「显式路径不校验扩展名，将以 `video/mp4` 发送，provider 可能拒绝」，
  `MIME` 行也会标 `(推断)` —— `/videotest` 照样准备成功（正常消息流由闸门 ⓪ 筛选）。
- **配置被自动修正过时**会多出 `配置提示:` / `配置修正:` 行（与启动日志的两类
  warning 同源），写出「你写的原值 → 生效值」。`compress_max_mb` 只会是**一条**：
  `compress_max_mb=15.0 超过体积闸门 max_size_mb=13.0，已按闸门的 90% 收敛为 11.7`
  —— 一个旋钮只有一个生效值（0.2.1 曾先夹到 13.0、再降到 11.7，打出两条都在说
  "已修正"的提示，而生效值等于哪一条都没写全）。

---

## 兼容性说明（「我会永远陪着你」）

本插件经过针对 `astrbot_plugin_private_companion` 的专项审计，遵守以下约束：

| 约束 | 原因 |
|---|---|
| **不碰** `image_urls` / `audio_urls` | 陪伴插件主动管理这两个字段，核心还会据此切换 provider |
| **不碰** `prompt` / `system_prompt` | 陪伴插件大量读写 |
| **不替换** `extra_user_content_parts` 列表 | 只就地 `append`，且每次重新读取属性（陪伴插件会替换整个列表） |
| **不删除** `[Video Attachment: ...]` 占位文本 | 默认 `hide_local_path=true` 时会把它**改写成不含路径的说明**（等价于替换）；若下游插件依赖这个前缀做判断，请关闭 `hide_local_path` |
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
   └─ 读文件 → base64 → data:video/mp4 URI
   ↓ 分支 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ┃ gemini_native                  ┃ openai_compat
   ┃ 追加 AudioURLPart.mark_as_temp() ┃ POST {base_url}/chat/completions
   ┃ ↓                               ┃   image_url=data:video/*;base64,...
   ┃ provider → process_audio_url     ┃ ↓ 视觉模型返回文字理解
   ┃ → Part.from_bytes(video/mp4)     ┃ 追加 TextPart(分析结果).mark_as_temp()
   ┃ ↓                               ┃ ↓
   ┗━━━► 主对话请求带上视频/理解结果，模型作答
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
├── .github/workflows/ci.yml  lint（ruff 0.15.22）+ pytest 2 平台 × 4 版本
├── tests/                  pytest 套件（stubs/ 里是 astrbot 的最小桩）
├── core/                   与 AstrBot 解耦的纯逻辑（便于日后上移核心）
│   ├── config.py           配置模型与容错
│   ├── decide.py           「会发生什么」的唯一真相源（纯函数：闸门/策略/截断/分类）
│   ├── diagnostics.py      `/videotest` 的报告、启动日志与用法文本（纯渲染，零 AstrBot）
│   ├── media.py            ffmpeg/ffprobe 封装、压缩、缓存
│   ├── injector.py         执行 decide 的 MediaPlan（压缩编排 + data URI 编码）
│   ├── openai_vision.py    OpenAI 兼容视觉服务客户端（实测格式见模块说明）
│   └── placeholder.py      占位文本解析
└── hooks/
    └── llm_request.py      on_llm_request 注入逻辑
```

`core/` 下除 `injector.py` 从 `astrbot.api` 取 `logger` 之外**不依赖** AstrBot 的
任何符号（插件市场安全检查要求日志记录器必须且只能来自 `astrbot.api`，所以这一处
是有意保留的）。其余纯逻辑（`config` / `decide` / `diagnostics` / `media` /
`openai_vision` / `placeholder`）可以直接搬进 AstrBot 核心。

### CI

`.github/workflows/ci.yml` 两条门禁：

| job | 内容 |
|---|---|
| `lint` | `ruff check`，**pin 到 `0.15.22`**（AstrBot 自己 pin 的版本）。不 pin 就没有门禁：同一个仓库 `ruff 0.17.x` 的默认规则集会报 31 个 error |
| `test` | `pytest -q --junitxml=…`，矩阵 `{ubuntu-latest, windows-latest} × {3.10, 3.11, 3.12, 3.13}`，单 job 15 分钟超时，junit 报告作为 artifact 上传 |

CI 里设了 **`VV_REQUIRE_FFMPEG=1`**：缺 ffmpeg 时相关用例**失败**而不是静默 skip
——否则约 40 例会悄悄消失，CI 照样绿。本机跑测试时可用 `VV_FFMPEG_DIR=<目录>`
指定 ffmpeg 所在目录（不设则从 `PATH` 找）。

---

## 已知限制

- 视频以 **inline base64** 传输，体积受超时预算约束（约 7-9s/MB 上行）；
  中转站若提供 File API 可显著放宽（当前实测的中转站没有）。
- 编码后的 data URI 有 **18MB 硬上限**（对应约 13.5MB 原始体积），超过会直接
  拒绝并给出原因，不会静默截断，也不会把上百 MB 的 base64 塞进请求。
- openai_compat 模式：视觉模型必须是**真能看视频**的模型（如 gemini-3.x-flash 系）；
  主模型拿到的是视觉模型的文字理解而非原片。
- gemini_native 模式：仅支持 **Gemini 系** provider（依赖其按 MIME 构造媒体 Part 的行为），
  且模型必须声明 **audio** 模态；走 OpenAI 协议的 Gemini 代理需在
  `additional_provider_types` 中显式放行。
- 不支持多视频并行压缩；单条消息的视频串行处理（openai_compat 的视觉调用是并发的）。
- 压缩缓存按源文件内容哈希做 key，**单个产物大于 `cache_max_mb` 时不会进缓存**，
  这类视频每次都会重新压缩。0.2.1 是**直接跳过它**（缓存里的其它条目不受影响）；
  0.2.0 相反：会先放进去、再被自己挤掉，还会连带清空其它条目。
- `/videotest` 只检查"路径存在且是文件"，**不校验扩展名**：给什么文件它就把什么
  按视频编码并发送（openai_compat 下会真的 POST 出去）。0.2.1 起报告里会有
  `提醒   :` 行 + `MIME   : … (推断)` 明说这件事，但**行为不变**（不会改成拒绝）。
  仅管理员可用，请勿在生产环境拿它测试非视频文件。
- 注入给模型的内容（原生视频块 / 视觉分析文字）都标记为**非持久化**，不会写进对话历史；
  因此隔一轮再追问视频细节时，模型手上已经没有上一轮的素材。

## 许可

MIT
