# 基米看得见视频 · astrbot_plugin_video_vision

> 当前版本 **0.3.0**（版本号的唯一来源是 `metadata.yaml`；`main.py` 不再持有副本）。

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
- **ffmpeg 并发限流**：进程内**全局**上限（默认 2），排队超时降级为 `busy`；
  不再是"最多 6 个 ffmpeg 一起抢 2 个核"（原上限只是默认线程池的 `min(32, cpu+4)`）
- **视觉结果缓存（仅 `openai_compat`）**：同一视频 + 同一提问 + 同一模型/prompt 的**重复投递**不再重付一次上传 + 推理（TTL 600s / LRU 64 条，纯内存）；`/videotest` **绕过**缓存
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
  清洗自 4.23.5 起）。**这个下界是按符号存在性核对出来的，没有在 4.24.1 上
  实跑过** —— 真机全流程只在 4.29.1 上跑过。若你在 4.24.1–4.28.x 上遇到问题，
  请带 `/videotest` 输出回报 issue。

---

## 配置项

| 键 | 默认 | 说明 |
|---|---|---|
| `enable` | `true` | 总开关 |
| `provider_mode` | `openai_compat` | `openai_compat`（默认）/ `gemini_native`。视觉服务三项没配齐时**自动降级为 `gemini_native`**，因此老配置直接升级不会失效 |
| `openai_base_url` | `""` | 视觉服务**完整接口地址**，写全到 `/chat/completions` 或 `/responses`（按结尾自动选协议）；只填到 `/v1` 兜底前者 |
| `openai_api_key` | `""` | 视觉服务密钥。面板里是**密码框**（schema `secret: true`），日志只打后 4 位 |
| `openai_model` | `gemini-3.8-flash-high` | 视觉模型，须支持视频 |
| `openai_timeout_sec` | `180` | 视觉服务超时（上传约 7-9s/MB）。既是单阶段超时，也是单个视频的整体硬上限 |
| `openai_max_tokens` | `2048` | 视觉回答 token 上限 |
| `vision_with_question` | `true` | 把用户随视频附带的文字一起发给视觉模型 |
| `vision_describe_prompt` | 见 schema | 无附带文字时的通用描述提示词（长文本，面板里是多行文本框） |
| `vision_qa_prompt` | 见 schema | 带问题提示词模板，`{question}` 会被替换（长文本，面板里是多行文本框） |
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
| `notify_on_failure` | `true` | 失败时追加一句简短说明给模型（`[Video attachment was skipped: …]`；"太长被拒"与"太大被拒"是两个不同的短语，见下） |
| `max_compress_seconds` | `45.0` | 压缩耗时**墙钟**上限（超时终止整棵进程树），超时则降级 |
| `ffmpeg_path` | `""` | 留空则从 PATH 查找；也可填所在目录 |
| `cache_enabled` | `true` | 启用压缩缓存 |
| `cache_max_mb` | `512` | 缓存上限，超出按 LRU 淘汰（下限 16；**单个产物大于此值时不进缓存**） |
| `ffmpeg_max_concurrency` | `2` | 进程内**全局**（跨会话）同时进行的 ffmpeg 转码数（0-8）。**`0` = 关闭限流**，行为回到 0.2.1 |
| `ffmpeg_queue_timeout_sec` | `30.0` | 单个视频**等待槽位**的上限（1-600，不含转码本身）。超时 -> 失败分类 `busy` |
| `ffmpeg_singleflight` | `true` | 同一内容 + 同一组压缩参数的并发转码只做一次（**依赖 `cache_enabled`**） |
| `vision_cache_enabled` | `true` | 视觉理解结果缓存。**只在 `openai_compat` 生效**（`gemini_native` 不查不写），**只对完全相同的重复投递命中**（视频标识 = 源 + 压缩参数，再加**声明 MIME**、最终 prompt、端点、模型、max_tokens、**API key 指纹**），纯内存、重启即失效 |
| `vision_cache_ttl_sec` | `600.0` | 条目存活时长（60-86400）。命中**不**续期 |
| `vision_cache_max_entries` | `64` | 条目上限（1-1024），超出按 LRU 淘汰；64 条约 1MB |
| `debug_log` | `false` | 详细调试日志（缓存计数只在这个级别可见；限流摘要在 `/videotest` 真的排过队时也会多一行） |

> **上面 6 个 0.3.0 新键填错类型会告警（只一次）**：`"false"` / `"0.5"` / `"abc"`
> 这类值过去被**静默**转换 —— `ffmpeg_max_concurrency="false"` → **2**（限流反而
> 开着）、`"0.5"` → **0**（限流被关掉）、`vision_cache_enabled="abc"` → **False**
> （缓存被关）。现在生效值**不变**，但启动日志会多一条 `配置提示:`，把**你填的原值**
> 与**实际生效值**写在同一句里；同一个「键名 + 原值」在进程内只提示一次，不刷屏。

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

每处理一个视频会在默认级别打一条 INFO，**两个数字含义不同**：

```
[VideoVision] 视频已备好: a.mp4 | compressed | 15.28MB -> 0.92MB | 时长 8.0s | 本地处理 493ms | 正常
```

- `时长` 是**视频本身**多长（决定上传体积与时长上限）；
- `本地处理` 是这条消息为它花的本地时间（探测 + 压缩 + 编码），**不含排队**；
  排队等待另有独立日志（≥1s 才打一条 `ffmpeg 排队 2.5s（limit=2，前方 1 个）`）。
  小于 1 秒按毫秒渲染（`493ms`），不会出现「本地处理 0s」。

---

## 并发、排队与缓存

### 先说清楚"原来的并发上限"是什么

`asyncio.to_thread` 的实现是 `loop.run_in_executor(None, …)` —— 用的是**全进程
共享**的默认 `ThreadPoolExecutor`，大小 `min(32, cpu+4)`（2 核小机器 = **6**、
16 核 = 20）。AstrBot 核心自己就在这个池里跑图片/音视频转码、TTS、自托管 ASR
与工具调用。所以：

- 2 核机器上"3 个群同时发视频"的真实后果是**最多 6 个 ffmpeg 同时跑**；
- 单个 ffmpeg 在 2 核上只用掉约 **44%-77%** 的 CPU（本机实测 3 个 18MB/18s 源：
  median 43.9% / 50.8% / 77.4%，p95 69% / 85% / 89%）—— 6 个并发是**明显超订**，
  外加 6 份源文件与 base64 常驻（13MB 源 → 17.33MB 字符串）；
- 上限**不是"无限"**，但也从来没人按核数算过它。

`ffmpeg_max_concurrency` 就是把这个"隐式的池上限"换成**显式的、按核数选过的**数字。

### 2 核实测（`SetProcessAffinityMask(0x3)` 钉核 + 6 线程默认执行器模拟 2 核 AstrBot）

本机（Intel Ultra 7 255H / 16 逻辑核）把**进程树**钉在 2 个逻辑核上（子进程继承），
默认执行器设成 6 线程（= 2 核机器的 `min(32, cpu+4)`），8 个**内容各不相同**的
720p/18s 源（每份约 18MB）同时发，统一走压缩路径（`max_size_mb=2.0`、
`compress_max_mb=1.8`、CRF 28），真 ffmpeg，每组 3 次取中位：

| 限流 | n=8 总耗时 | 单个 ffmpeg 的槽内时长（中位/最大） | 端到端等待（中位/最大） | 排队（中位） | 峰值 ffmpeg |
|---|---|---|---|---|---|
| 不限（池上限 6） | 14.91s | 11.70 / 14.91s | 11.70 / 14.91s | 0s | 6 |
| **2（默认）** | **15.09s（+1.3%）** | **3.77 / 3.95s** | **9.32 / 15.09s** | 5.47s | **2** |
| 1 | 18.19s（+22%） | 2.28 / 2.41s | 10.15 / 18.19s | 7.91s | 1 |

读法（**别读错口径**）：

- **总吞吐基本不变**（15.09 vs 14.91s，+1.3%）；**`limit=1` 才是真的变慢**
  （+22%）—— 单个 ffmpeg 用不满 2 个核（见上），串行等于让一半机器空转。
  所以默认值是 **2**：两个槽位大体把 2 核喂满；不是 1，也不是 `cpu_count` 公式
  （配置默认值必须可预测、可文档化、可被测试逐字校验）。
- **峰值进程 6 → 2**；**单个 ffmpeg 的槽内时长 11.70 → 3.77s（约 3 倍）**
  —— 这两条才是限流买到的资源确定性（内存峰值同样按比例下降）。
- **端到端等待：中位 11.70 → 9.32s（−20%），最大 14.91 → 15.09s（不变）**。
  所以"尾延迟降 64%"这种说法**不成立**：那个数字只统计**拿到槽之后**的时长
  （就是上表的"槽内时长"列），不是用户实际等的时间。用户实际最多仍要等一个
  makespan；限流不会让最慢的那个人更快，只是让**没有人**被迫和另外 5 个 ffmpeg
  一起抢 2 个核。
- 代价：**高峰期单条请求最多多等 `ffmpeg_queue_timeout_sec`**（默认 30s）。
  排队超时的失败分类是 **`busy`**（注入给模型的话是
  `the server is busy processing other videos`）—— 它**不是** `timeout`，
  别去调 `max_compress_seconds`，该调的是并发上限或本项。

### 排队发生在事件循环侧（worker 跑在 ffmpeg 专属执行器）

`core/limits.py` 的 `SlotLimiter` 用 `asyncio.Queue` 预填 `limit` 个令牌，
钩子里是：

```python
async with limiter.slot(name=name) as slot:
    # 取到槽 -> 提交到 ffmpeg 专属执行器 -> worker 真正结束才还槽
    outcome = await slot.wait_for_worker(limiter.submit(prepare_video, ...))
```

**`wait_for_worker` 不是装饰**：外层被取消（框架超时、用户撤回）时，槽位必须留在
**还在跑的 ffmpeg 线程**手里，由它的 done-callback 归还。少了这一层，`async with`
会当场还令牌，而 ffmpeg 子进程还在跑 —— 真实并发于是越过 `limit`（实测 `limit=1`
峰值 **2**）。照抄时把提交**包在** `wait_for_worker` 里，不要只 `await` 提交结果。

**没有**把 `threading.Semaphore` 塞进 `prepare_video`：那样 2 核机器的 6 个默认
线程里会有 4 个在"睡觉"，把核心的 media/TTS/ASR 一起饿死。提交走
`SlotLimiter.submit` —— `max_workers == limit` 的**专属**执行器，不是
`asyncio.to_thread` 那个**全进程共享**的默认池：池被核心占满时，后者会让任务躺在
池队列里无限期排队，而槽位已经被取走（`wait_for_worker` 对 worker 的等待没有
超时）。`tests/test_limits.py` 用 AST 锁住"每个阻塞提交都在槽内"，并专门测了
取消/超时**不吞令牌**（槽位永久 -1 会让全站慢慢卡死）。

- 排队 INFO 与超时 WARNING 都带 **视频名**（`slot(name=…)` → `视频=a.mp4`），
  线上能直接定位"是哪条视频排了 28 秒"；不带名字时不留空尾巴。
- **小于 1 秒一律渲染成毫秒**（`10ms` / `400ms`）：排队摘要、超时文案、缓存命中的
  「TTL 剩 Xs」与 `/videotest` 的报告共用同一个 `format_seconds()`，不会再出现
  「排队等待转码超过 0s」或「TTL 剩 0s」。`ffmpeg_queue_timeout_sec` 的配置下限是
  **1.0s**，所以排队这条最常见的是 `1.0s` / `12.3s`；亚秒形态出现在缓存 TTL 剩余
  与直接构造的边界用例里。

### 同一文件并发只转码一次（`ffmpeg_singleflight`）

两个群同时发**同一段视频**时，0.2.1 会各压一次（实测 2 次 `ffmpeg`、6 次
`ffprobe`、2 份 base64）。现在：键 = 压缩缓存已有的那个 key（内容哈希 |
`compress_max_mb` | `crf` | `preset` | `truncate_to`），临界区同时包住
`compress_video` 与 `cache.put`，follower 醒来后直接复用 leader 已发布的缓存产物。

- **只依赖 `cache_enabled`**：缓存关掉就没有可共享的产物，这时启动日志会有一条
  **WARNING**（配置自相矛盾：开了 singleflight 却没有缓存），每个视频再打一条
  debug 说明"本次已跳过"；两个线程各压一次（与 0.2.1 相同）；
- leader 失败时分流：确定性失败（`not_found` / `duration` / `corrupt` /
  `empty` / `too_large`）**原样返回**（重试只会把 45s 超时乘以 N）；
  瞬时失败（`timeout` / `exit_code` / `other`）→ follower **自己接管一次**；
- **等待超时不等于接管**（0.3.0 定稿语义）：follower 等满自己的等待预算后只是
  **放弃这一轮等待**，下一轮仍等同一个 leader —— 同一个 key 的并发恒为 1。
  同时打一条 `on_overdue` **WARNING**，让"leader 慢到超过 follower 预算"这件事
  在日志里可见（否则用户只会看到"卡住"）；
- 两个线程拿到的是**同一个缓存产物**（`used_path` 相同），不是各压一份。

### 视觉理解结果缓存（`vision_cache_enabled` / `vision_cache_ttl_sec` / `vision_cache_max_entries`）

> **只在 `openai_compat` 生效**：`gemini_native` 走的是「把视频块交给主 provider
> 亲眼看」，根本不产生可缓存的文本结果 —— 那条路径既不查也不写缓存
> （实测投递一次后 `hits/misses/stores` 全 0、`cache_len` 仍为 0）。

**先说清楚它不解决什么**：对同一段视频**换个问题再问**，不会因为缓存变快 ——
用户追问时如果不重发/不引用视频，插件的钩子根本不运行（历史里既没有视频占位符，
注入的分析也是 `mark_as_temp()` 的、落库时被丢弃）；追问时重新引用视频通常
**换了提问**，那就**不该**命中缓存（用上一问的答案回答这一问才是 bug）。

它解决的是另一半：**同一视频 + 同一提问 + 同一模型/prompt 的重复投递**
（同一个文件被两个人转发、失败重试、同一段视频再发一次）。这类投递**以前每次都要
重付一次上传 + 推理**（实测 60-120 秒、7-9s/MB 上行），**现在只付一次**
（第 2 次起服务端 `POST` 增量为 0）。

**API 计数证据**（服务端在本机 `http.server` 上自己数的 `POST` 次数，不是插件自报；
装配与 `tests/test_vision_cache.py` 相同：真 httpx + 真端点 + 真 ffmpeg + 真钩子）：

| 投递 | 变化维度 | 服务端 `POST` 增量 | 缓存条目 | 命中次数 | 注入文本里的服务端序号 |
|---|---|---|---|---|---|
| 第 1 次 | 视频 A + 提问 A | **+1** | 1 | 0 | `STUB#1` |
| 第 2 次 | **完全相同** | **+0** | 1 | 1 | 仍是 `STUB#1` |
| 第 3 次 | 提问 A → B | **+1** | 2 | 1 | `STUB#2` |
| 第 4 次 | 回到提问 A | **+0** | 2 | 2 | 仍是 `STUB#1` |
| `/videotest` 第 1 次 | 同一文件 | **+1** | 2（不变） | 2（不变） | `STUB#3` |
| `/videotest` 第 2 次 | **完全相同** | **+1** | 2（不变） | 2（不变） | `STUB#4` |
| 4 路**并发**完全相同 | 同时发起 | **+1**（其余 3 路等 leader 的结果） | 1 | 0 | 全是 `STUB#1` |

key 是**六个语义维度 + 两条身份维度**的 sha256（`KEY_VERSION=2`）：

```
v2 | 视频标识 | 声明 MIME | 最终 prompt | 协议 | 端点 URL | 模型 | max_tokens | API key 指纹
```

> **v2 为什么要多两条**（AUDIT-VV-R9 §2.2/§2.3，两条都实测过串味）：
> `视频标识` 只有"源哈希 + 压缩参数"，**不含发给端点的 MIME** —— 同一份字节改名
> `.mov` 后与 `.mp4` 共用一条缓存，第 2 次 **0 次 `POST`** 就拿到了 `.mp4` 那次的
> 答案；`api_key` 同理（换 key 后仍然命中旧答案）。现在少了任何一条都会 miss。

- **视频标识**（`VideoInjection.content_key`）＝ 模型实际看到的画面，取
  **「源 + 参数」而不是产物字节**：
  - 直通：`as-is:<源文件内容哈希>`；
  - 压缩/截断：`<action>:<压缩签名>`，签名 = 源内容哈希 + `compress_max_mb` +
    `crf` + `preset` + `truncate_to` —— 与**压缩缓存 key 完全同源**。

  为什么不用"真正发出去的字节"的哈希：同一源 + 同一组参数压出来的 mp4
  **每次都不一样**（熵编码 + 封装时间戳 + `-fs` 在 GOP 边界截断），实测 3 次
  同源同参 = 3 个不同的产物哈希。取产物哈希会让 `cache_enabled=false` 时
  视觉缓存**永不命中**（这正是 0.3.0 第一版的缺陷，已修）。压缩参数变了 =
  模型看到的画面变了 = 仍然必须换 key。
- **声明 MIME**：`guess_video_mime(used_path)`，也就是 data URI 里那个前缀
  （`data:video/mp4` vs `data:video/quicktime`）。它由**后缀**决定，所以"同一份
  字节另存为 `.mov` 再发"是一次新的声明 —— 端点若按 MIME 拒绝或差别对待，复用的
  旧答案会把失败掩盖掉。
- **最终 prompt**：`build_vision_prompt()` 的输出（已折叠用户问题 + 两个模板）。
  只哈希原始 question 会漏掉模板改动。
- **端点**：`(协议, 最终 URL)` —— 换端点/换协议回答会变。
- **模型 / max_tokens**：换模型=换回答；上限不同输出可能被截断。
- **API key 指纹**：`sha256(openai_api_key)[:12]`（48 bit）。**原文不进 key** ——
  载荷会进 `json.dumps`，明文多留一处就多一处泄漏面；而 12 位足够区分"换了 key"。
  为什么换 key 要换缓存：中转站（本项目文档承认的部署形态）会按 key 路由到不同
  后端/模型，换 key 之后命中旧答案，等于"用户修正了一个错 key，第一次重复投递
  还是拿到错 key 时的回答"。
- **不进 key**：API key 的**原文**、`openai_timeout_sec`、`vision_verify_ssl`、
  视频文件名（QQ 每次落地名不同，放进去等于永不命中）、会话 ID（跨会话共享正是
  收益来源）。

行为细节：

- **只缓存成功且非空的分析**：失败/超时绝不写缓存（否则一次网络抖动会被固定成
  TTL 内的持续故障）。
- **TTL 600s，命中不续期**；**LRU 64 条**（约 1MB 常驻）。纯内存、重启即失效、
  不跨进程。
- **与压缩缓存的耦合**：关掉 `cache_enabled` **不影响本缓存命中**（key 取「源 +
  压缩参数」，不依赖压缩产物），但需要压缩的视频**每次投递都要重新压一遍** ——
  本缓存省掉的是上传 + 推理，省不掉本地 ffmpeg。这个组合启动时会 **WARNING 一次**
  说明代价（不刷屏）；想连本地压缩一起省掉就保持 `cache_enabled=true`。
- 命中时**不调用**视觉服务，注入文本仍走原来的包装（视频名/模型名用本次的），
  日志单独打一条 `视觉缓存命中（TTL 剩 Xs）`（剩余 <1s 时是毫秒，如 `400ms`）——
  命中时 `elapsed/token` 记 0，不会伪装成"接口变快了"。
- **`/videotest` 绕过缓存**：诊断必须反映真实链路，那条路径直接调 `analyze_video`。
  实测同一文件连跑两次诊断 = **两次真实 `POST`**（上表 `+1 / +1`），报告在时间数字
  归一化后逐字节相同。**`/videotest` 不是受益场景**：它既不会写缓存，也不会因为
  前一条真实消息刚缓存过而少调一次。
- **同一 key 的并发投递只发一次请求**（P2-3）：4 路同时发同一视频 = **1 次真实
  `POST`**，其余 3 路等 leader 的结果（上表最后一行）。leader 失败时失败结果同样共享，
  但**不写缓存** —— 下一次投递会重新请求。它是"并发去重"，与串行命中是两条路径：
  前者不产生缓存条目、后者靠 TTL；`vision_cache_enabled=false` 时**两者都没有**。
- 命中日志里的剩余 TTL **小于 1 秒时按毫秒打印**（`TTL 剩 400ms`），不会出现「TTL 剩 0s」。
- `vision_cache_enabled=false` → **完全不建缓存对象**、`content_key` 也不算。
  **只有另外两个 0.3.0 开关也关闭、其余键保持 0.2.1 原值时**，注入决策与既有
  日志行为才与 0.2.1 等价（口径与那条唯一例外见下节）—— 只关这一个键，限流队列
  与 singleflight 仍然生效、仍然会打日志。

### 三条开关 = 回到 0.2.1

| 开关 | 关掉之后 |
|---|---|
| `ffmpeg_max_concurrency = 0` | 不建令牌队列、不加任何等待、日志零新增 |
| `ffmpeg_singleflight = false` | 压缩块不经过 flight，两个线程各压一次 |
| `vision_cache_enabled = false` | 不建缓存对象、不算 `content_key`、钩子不查不写 |

外加 TTL / 条目数 / 排队上限等参数项（`vision_cache_ttl_sec`、
`vision_cache_max_entries`、`ffmpeg_queue_timeout_sec`）—— 它们只在对应开关打开时
才被读取，关掉开关后不产生任何行为。

**三条全关之后到底等价到什么程度（按实测写，不写"逐字节"）**：
**注入决策与既有日志行为等价** —— 9 个素材 × 11 组配置的 198 条 `prepare_video`、
5 个素材 × 4 组配置的 40 份 `/videotest` 报告、3 次钩子端到端（共 **241 条观测**）
与 0.2.1（`c00f6a2`）逐字段比较 **0 差异**：`data_uri` 长度与全字节 sha256、
`action`、`size_mb`、`duration_sec`、`note`、`used_path` 的类别全部相同，
报告文本在时间数字归一化后相同，日志零新增；同树复跑对照同样 0 差异。
（注意：`data_uri` 的**全字节**相等只在"编码器输出确定"的预算下成立。把 `-fs`
压到极小、或素材特别短时，x264 自己就可能每次产出不同字节——那种情形下**同一棵树
连跑两次也不相等**，属编码器非确定性，与本插件的开关无关。）
**唯一的例外**是当旧键 `max_compress_seconds` 与队列超时预算冲突时新增的预算告警：

```
ffmpeg_queue_timeout_sec(30s)+max_compress_seconds(60s) 已吃掉视觉服务超时预算(openai_timeout_sec=180s) 的一半以上，建议调小其中一项（排队上限或压缩上限）
```

它是 0.3.0 `VideoConfig.normalized()` 新加的（`core/config.py`），**与这三个开关
无关**：三开关全关 + `max_compress_seconds=45`（0.2.1 默认）= 0 差异；把它改成 60
（仍是合法的 0.2.1 旧键值）就多这条告警 —— 启动日志多一条 `WARNING … 配置已自动修正`
且 `/videotest` 多一行 `配置修正:`。这条告警只在
`ffmpeg_queue_timeout_sec + max_compress_seconds >= openai_timeout_sec × 0.5`
且 `provider_mode` 实际为 `openai_compat` 时出现。

**除它之外还有两处"三开关全关也看得见"的修正**（都是 0.3.0 收官轮改的，与开关无关，
且都属"原来那句文案是错的"）：

1. **本地 IO 失败不再被说成"超限"**：中文 locale 下 ffmpeg 的 `拒绝访问。` 曾让
   `short_reason` 命中 `拒绝` 分支，用户看到 "it exceeds the size limit"（英文 locale
   得到正确的 "compression failed"）。现在权限/磁盘失败有专门文案
   （`local file access was denied` / `the local disk is full`），未知失败落回中性
   `compression failed`。**这句文案会出现在注入给模型的上下文里**，所以它是可观测差异。
2. **配置写错类型会有一条 WARNING**：`ffmpeg_max_concurrency="false"`（被当 2）、
   `"0.5"`（被当 0）、`vision_cache_enabled="abc"`（被当 false）以前**静默**吞掉，
   现在各打一条一次性 `配置提示:`，说明"你填的 X 被当成 Y"。生效值与 0.2.1 相同，
   只是不再无声。

`LimiterStats`（等待次数 / 最长等待 / 超时次数）与 `VisionCacheStats`（命中 /
未命中 / 写入 / 淘汰 / 过期）在 `debug_log=true` 时逐次写进日志。`/videotest` 只在
**真的排过队或被排队超时拒过**时多一行 `队列   :` 摘要（例如
`队列   : 排队 3 次，最长 12.3s，累计 20.0s，超时 1 次`）—— 标签与
`core/diagnostics.py` 的实现逐字一致（写成「排队」会读成「排队: 排队 3 次…」）；
正常路径的报告格式**一字不变**（`LimiterStats.summary()` 为空串就不渲染）。

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
结论   : 体积 21.8MB > 闸门 13.0MB → 压缩
prompt : （gemini_native 不调用视觉服务，无 prompt）

✅ 准备成功（0.64s）
动作: 已压缩
产物: 1.79 MB / 20.0 s
编码: data URI 2.38 MB
MIME   : video/mp4（压缩产物，固定为 mp4）
说明: 体积 21.8MB > 闸门 13.0MB -> 1.8MB

预计上传耗时: 约 14.3s（provider 预算 180s）
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
  与 `max_size_mb` 无关。`结论` 行已经把它算进来 —— 闸门调到 13.5MB 以上时，
  原本的"未超限，直接送"会变成提前说明"编码后约 XMB > 编码上限 18MB，实际会以
  `too_large` 失败"；压缩目标（`compress_max_mb`）超过同一阈值时也会在结论行提示。
- **时长读不出来**（`.mp4` 等视频容器 + 有 ffprobe 却解析失败）时，结论行直接写
  「无法解析（时长读取失败）→ 会以 corrupt 失败」，不再先说"未超限，直接送"、
  下一段再报 `分类: corrupt`。
- **扩展名不在已知视频容器内**时（如 `.gif`），结论行上方会多一行 `提醒   :`
  说明「显式路径不校验扩展名，将以 `video/mp4` 发送，provider 可能拒绝」，
  `MIME` 行也会标 `(推断)` —— `/videotest` 照样准备成功（正常消息流由闸门 ⓪ 筛选）。
- **结论行的标签不叫「判定」**：见下面「兼容性说明」里那条出站文本约束 ——
  0.2.1 的 `判定   : ` 会被陪伴插件当成"复核模型泄漏的内部判断"整行删掉。
- **配置被自动修正过时**会多出 `配置提示:` / `配置修正:` 行（与启动日志的两类
  warning 同源），写出「你写的原值 → 生效值」。`compress_max_mb` 只会是**一条**：
  `compress_max_mb=15.0 超过体积闸门 max_size_mb=13.0，已按闸门的 90% 收敛为 11.7`
  —— 一个旋钮只有一个生效值（0.2.1 曾先夹到 13.0、再降到 11.7，打出两条都在说
  "已修正"的提示，而生效值等于哪一条都没写全）。
- **失败分类**（`分类:` 行）是机器可读的：`corrupt` / `empty` / `not_found` /
  `duration` / `too_large` / `timeout` / `exit_code` / `missing` / `busy`，以及配置
  拒绝的三条 —— `rejected_size`（体积超闸门）/ `rejected_duration`（时长超上限）/
  `rejected_size_and_duration`（两者都超）。注入给模型的那句话按分类取
  （`it exceeds the size limit` / `its duration exceeds the limit` / …）：
  **"太长被拒"不会被告知"它太大了"**，否则模型会去劝用户压体积。`busy` 是
  "排队等槽位超时"（`the server is busy processing other videos`），与 `timeout`
  （ffmpeg 自己超时）是两回事。
- **`准备成功（X s）` 的耗时含排队时间**（计时起点在取槽之前）：高峰期这个数字
  会比 ffmpeg 实际耗时大，差值就是排队。`/videotest` 与真实消息走**同一个**
  `ffmpeg_max_concurrency` 槽位，所以诊断结果与线上一致。

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
| **出站文本不长得像复核字段** | 陪伴插件在**发出去之前**逐行删掉"复核模型泄漏的内部判断"。它的 `user_memory.py:9174-9210` 认 `(?:判定\|审核\|复核)(?:结果\|结论\|原因)?\s*[:：]`，`:9212-9235` 丢掉命中的行，`main.py:10677-10700` 把整条回复换成清洗后的正文。0.2.1 的 `/videotest` 报告里 `判定   : …` 因此整行消失（真机日志 21:00:39 / 11:34:53 两条 ERRO `reason=复核模型输出判定字段`；21:00:39 实际发出的正文里既没有那一行，闸门行的缩进也被一起去掉）。现在标签是 `结论   : `，`② modalities …` 两行也不再出现「无法校验/清洗」这种成对措辞；`tests/test_interop.py` 把那份守卫逐字复刻后逐行断言 |

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
│   ├── limits.py           ffmpeg 并发限流（async 令牌）+ 同文件 singleflight（线程事件）
│   ├── openai_vision.py    OpenAI 兼容视觉服务客户端（实测格式见模块说明）
│   ├── vision_cache.py     视觉结果缓存（TTL + LRU，纯内存，零 AstrBot）
│   └── placeholder.py      占位文本解析
└── hooks/
    └── llm_request.py      on_llm_request 注入逻辑
```

`core/` 下除 `injector.py` 从 `astrbot.api` 取 `logger` 之外**不依赖** AstrBot 的
任何符号（插件市场安全检查要求日志记录器必须且只能来自 `astrbot.api`，所以这一处
是有意保留的）。其余纯逻辑（`config` / `decide` / `diagnostics` / `media` /
`limits` / `vision_cache` / `openai_vision` / `placeholder`）可以直接搬进 AstrBot 核心。

### CI

`.github/workflows/ci.yml` 两条门禁：

| job | 内容 |
|---|---|
| `lint` | `ruff check`，**pin 到 `0.15.22`**（AstrBot 自己 pin 的版本）。不 pin 就没有门禁：同一个仓库 `ruff 0.17.x` 的默认规则集会报 31 个 error |
| `test` | `pytest -q --junitxml=…`，矩阵 `{ubuntu-latest, windows-latest} × {3.10, 3.11, 3.12, 3.13}`，单 job 15 分钟超时，junit 报告作为 artifact 上传 |

CI 里设了 **`VV_REQUIRE_FFMPEG=1`**：缺 ffmpeg 时相关用例**失败**而不是静默 skip
——否则约 40 例会悄悄消失，CI 照样绿。本机跑测试时可用 `VV_FFMPEG_DIR=<目录>`
指定 ffmpeg 所在目录（不设则从 `PATH` 找）。

### 契约用例（没有 `@register` 也能被识别）

`tests/test_contract.py` 会起一个**子进程**跑 `tests/contract_probe.py`，在**真
AstrBot** 里验证：导入期零 `DeprecationWarning`、`star_map` 里登记的类型就是
`VideoVisionPlugin`、`on_llm_request` 恰好 1 条且 priority 仍是 `-300000`、
`/videotest` 仍登记为 `cmd_videotest`。

* venv 里没装框架 → 探针退出码 2 → 用例 **skip**（CI 不会因此变红）；
* 想真跑，用装了框架的解释器，并用 `VV_ASTRBOT_APP` 指向它的 `app` 目录
  （**仓库里不写任何开发机私有路径**）：

```bash
VV_ASTRBOT_APP=/path/to/AstrBot/backend/app python -m pytest tests/test_contract.py -q
```

0.3.0 已用 AstrBot **4.29.1** 实测过（同一套断言）：

```
import_deprecation_warnings = []
star_map_key = 'data.plugins.astrbot_plugin_video_vision.main'
star_cls_type_is_plugin = True
on_llm_request_handlers = [('on_llm_request', -300000)]
command_handlers = ['cmd_videotest']
plugin_id = josephtian876/astrbot_plugin_video_vision | PLUGIN_VERSION = 0.3.0
checks = {'title': True, 'verdict_line': True, 'prepare_ok': True, 'mime_line': True}
```


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
- 不支持多视频串行以外的压缩调度：**同一条消息里的视频仍然是一个一个处理**
  （跨会话才有并发，并受 `ffmpeg_max_concurrency` 约束；openai_compat 的视觉
  调用是并发的）。高峰期单条请求最多多等 `ffmpeg_queue_timeout_sec`。
- **限流与缓存都只在进程内**：`ffmpeg_max_concurrency` 是**本进程**的闸门，
  不是全局资源锁；多开 AstrBot 实例时机器上仍会有 `实例数 × limit` 个 ffmpeg。
  视觉结果缓存同理 —— 纯内存、重启即失效，不跨进程共享。
- **视觉结果缓存只在 `openai_compat` 生效**，且**只对完全相同的重复投递**命中
  （视频字节 + 最终 prompt + 端点 + 模型 + max_tokens）；命中时日志里的
  `elapsed/token` 记 0，别拿它当"接口变快了"的证据。`/videotest` **绕过**缓存。
- **0.3.0 起不再使用 `@register`**（AstrBot 4.29.1 的 `core/star/register/star.py`
  文件头就标着 `[DEPRECATED]`，且首次使用会发 `DeprecationWarning`）：插件身份
  **只**来自 `metadata.yaml`。它整个丢失时插件仍能加载，但面板/日志里的身份会
  退化成 `unknown/unknown` —— 这是删装饰器的唯一回归面，已用测试锁住四个必填字段。
- 压缩缓存按源文件内容哈希做 key，**单个产物大于 `cache_max_mb` 时不会进缓存**，
  这类视频每次都会重新压缩。0.2.1 是**直接跳过它**（缓存里的其它条目不受影响）；
  0.2.0 相反：会先放进去、再被自己挤掉，还会连带清空其它条目。这类产物也**无法**
  被 singleflight 共享（没有可共享的位置），等待者会自己再压一次。
- `/videotest` 只检查"路径存在且是文件"，**不校验扩展名**：给什么文件它就把什么
  按视频编码并发送（openai_compat 下会真的 POST 出去）。0.2.1 起报告里会有
  `提醒   :` 行 + `MIME   : … (推断)` 明说这件事，但**行为不变**（不会改成拒绝）。
  仅管理员可用，请勿在生产环境拿它测试非视频文件。
- 注入给模型的内容（原生视频块 / 视觉分析文字）都标记为**非持久化**，不会写进对话历史；
  因此隔一轮再追问视频细节时，模型手上已经没有上一轮的素材。

## 许可

MIT
