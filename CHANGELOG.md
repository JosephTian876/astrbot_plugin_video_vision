# 更新日志

## 0.1.2

- 启动日志打印的版本号改为从 `metadata.yaml` 读取，不再硬编码，避免与发布版本脱节

## 0.1.1

按插件市场自动安全检查（LLM Guard）意见修复日志规范。**无功能变更。**

- `hooks/llm_request.py`：日志记录器统一改为从 `astrbot.api` 导入
- `core/injector.py`：移除对 Python 内置日志模块的引用，改用 `astrbot.api` 的 logger
- `main.py`：移除导入失败时回退到内置日志模块的分支，直接使用 `astrbot.api` 的 logger

规范要求日志记录器**必须且只能**从 `astrbot.api` 导入，不得直接使用 Python 内置
日志模块。现已全插件无任何内置日志模块引用，并已全仓库复查确认。

安全检查结论为 `malicious=0, suspicious=0` —— 功能实现本身干净，仅日志规范问题。

## 0.1.0

首个版本。

- 在 `on_llm_request` 阶段把视频转成内容块注入请求，使 Gemini 以 `inline_data`
  **原生收到视频**（而非抽取关键帧），从而获得完整的画面、动作、字幕与时间轴理解能力
- **兼容 QQ / NapCat 把视频作为「文件」投递**的情况（核心此时产出
  `[File Attachment: ...]` 而非 `[Video Attachment: ...]`）
- 注入成功后抹除占位文本中的本机路径，避免诱导模型自行用代码工具解析文件
- 视频块标记为**非持久化**，不会写入对话历史、也不会在后续轮次被重发
- 超出体积闸门时自动用 ffmpeg 压缩（受约束的 CRF，画质随画面复杂度自适应）
- 压缩结果按源文件内容哈希缓存；超长视频自动截断
- 全链路失败降级：视频处理出错**绝不影响**正常文字对话
- 附带 `/videotest` 诊断指令，逐步输出体积、时长、压缩动作与预计上传耗时

### 兼容性

需要 **AstrBot >= 4.26**：

- `ContentPart.mark_as_temp()` / `dump_messages_with_checkpoints()` 自 4.24.5 起提供
- 核心产出 `[Video Attachment: ...]` 占位符自 4.26.0 起

已在 **aiocqhttp（QQ / NapCat）** 上实测通过。
