"""给「陪伴」宿主（``astrbot_plugin_private_companion``）的只读状态接口。

契约来源：鸭子类型，不是 AstrBot 官方字段
==========================================

宿主在 ``astrbot_host/integration_registry.py:192-219`` 这样读第三方插件::

    api = getattr(getattr(star, "star_cls", None), "extension_api", None)
    method = getattr(api, "companion_v2_status", None)

``extension_api`` / ``companion_v2_status`` 在 AstrBot 里**都不存在**：
``core/star/base.py`` 的 ``Star`` 与 ``core/star/star.py`` 的 ``StarMetadata``
都没有这两个名字，加载器 ``core/star/star_manager.py`` 也没有属性白名单（它只
``setattr`` 三个身份字段）。所以这是**约定**而不是官方机制：双方各持一份字段名与
字面量，靠契约测试对齐，谁都不 import 对方的模块。

四道校验（逐条对应宿主行号，本模块的形状必须同时满足）
======================================================

1. ``:194`` 方法必须可调用，且 ``inspect.iscoroutinefunction`` 为假 ——
   写成 ``async def`` 会让宿主把它当成"还没有状态适配器"（``installed``）；
2. ``:198-202`` 返回值若是 awaitable 会被 ``close()`` 后抛 ``ValueError``
   （``degraded``），所以本方法**必须同步返回**；
3. ``:203`` ``type(status) is dict``（严格类型比较：``dict`` 子类、``Mapping``
   代理、模型实例都不行），且 ``schema`` 逐字等于
   ``companion.provider-status.v1``；
4. ``:204-206`` ``plugin_id`` 逐字等于宿主 catalog 里的登记名（**裸目录名**，
   不是 AstrBot 注入的 ``author/name`` 形态）；``enabled`` 必须是 ``bool``
   （``type() is bool``，``1`` / ``"true"`` 都不行）；``runtime_ready``
   是 ``None`` 或 ``bool``。

为什么只有四个字段
==================

宿主 ``:203-207`` 只校验这四个字段的**形状**，``:210`` 只把 ``status_connected``
/ ``enabled`` / ``runtime_ready`` ``update`` 进清单；``:208-209`` 的注释原文是
``Only these typed scalar fields can cross into the manifest. No raw
configuration, private error text, paths, tokens or user data.`` 多给的键既不
进清单、也违反那句话的意图，将来宿主收紧校验时还会直接变成 ``degraded``。
所以本模块**只回这四个标量**：``schema`` / ``plugin_id`` / ``enabled`` /
``runtime_ready`` —— 没有配置原文、没有 API key 指纹、没有缓存路径、没有错误
文本、没有插件版本号（版本号宿主自己从 ``metadata.yaml`` 读，见 ``:169``）。

为什么是实例属性而不是类属性
============================

宿主读的 ``star_cls`` 是**插件实例**（``star_manager.py:1230-1246`` 先
``star_cls_type(...)`` 造实例、再 ``setattr(metadata.star_cls, "plugin_id", …)``），
而 ``enabled`` / ``runtime_ready`` 只能来自**运行中**的 ``self.config`` ——
类属性拿不到配置。插件被停用/卸载时 ``star_cls`` 为 ``None``
（``star_manager.py:1248``），宿主在更早的 ``activated`` 检查处就返回
``disabled``，根本不会走到这里。
"""

from __future__ import annotations

from typing import Any

#: 宿主定义的契约版本（``astrbot_host/integration_registry.py:27``）。
#: 这里**复制**一份字面量而不是 import：本插件不能依赖宿主的包结构，宿主也不能
#: 依赖本插件的；两边各持一份常量，靠 ``tests/test_status_api.py`` 与宿主的用例
#: 共同守住"两边必须逐字一致"。
STATUS_SCHEMA = "companion.provider-status.v1"

#: 宿主 catalog 里的登记名 —— AstrBot 的**插件目录名**。
#: 绝不能写成 ``self.plugin_id``：``star_manager.py:1244-1246`` 会把它覆盖成
#: ``author/name`` 形态（``core/star/star.py:79-82``），而宿主是逐字比较。
PLUGIN_ID = "astrbot_plugin_video_vision"


def mode_is_usable(config: Any) -> bool:
    """``effective_provider_mode`` 的依赖满足了吗 —— "实际能不能干活"。

    语义刻意**不是**"用户配置的模式被原样采纳"：``provider_mode=openai_compat``
    但三要素没配齐时，``VideoConfig.effective_provider_mode`` 会把生效模式降级成
    ``gemini_native``，而这条降级路径本来就能工作（Gemini 主 provider 亲眼看
    视频）。所以**降级但可用仍算就绪** —— 否则用户配置写错三项，宿主清单会一直
    挂在"尚未就绪"，可真实请求却每次成功。

    - ``openai_compat``：依赖全在插件自己的配置里（``base_url`` / ``api_key`` /
      ``model``），直接读 ``VideoConfig.openai_service_ready``；
    - ``gemini_native``：唯一的依赖是**宿主当前的 Gemini provider**。那是宿主
      运行状态，钩子与状态接口都读不到（``resolve_provider`` 只能在请求时
      ``await``），所以这里按"插件侧没有必需配置项"认定为满足。真正不满足时，
      注入路径会按既有的 provider / 模态闸门跳过并给出日志，不会假装看见视频；
    - 取不到配置或模式值不认识：按未就绪报（宁可让宿主显示 ``waiting``，也不
      谎报能用）。

    纯函数：只读配置对象，不 I/O、不 await、不打日志。
    """
    if not bool(getattr(config, "enable", False)):
        return False
    mode = getattr(config, "effective_provider_mode", "")
    if mode == "openai_compat":
        return bool(getattr(config, "openai_service_ready", False))
    return mode == "gemini_native"



class VideoVisionStatusAPI:
    """一个只读、无副作用的运行状态门面。

    宿主每次刷新能力清单时调用一次 :meth:`companion_v2_status`；本类**不做**
    任何 I/O、不加锁、不 await。宿主把整个调用包在 ``try/except`` 里
    （``:217-218``），任何异常都会让它显示 ``degraded`` —— 所以可预期的"没有"
    （例如取不到 ``config``）在这里自己按未就绪报，不要把异常抛给宿主：状态
    接口自己坏掉才该是 ``degraded``。
    """

    def __init__(self, plugin: Any) -> None:
        """只保存插件对象的引用（只读，绝不修改它）。

        Args:
            plugin: ``VideoVisionPlugin`` 实例。挂载点在
                ``main.VideoVisionPlugin.__init__`` 的末尾，此时 ``self.config``
                已经就绪。
        """
        self._plugin = plugin

    def companion_v2_status(self) -> dict[str, Any]:
        """返回宿主能安全消费的四个标量字段。

        ``enabled`` 来自插件总开关 ``config.enable``；``runtime_ready`` 表示
        **当前配置下插件实际能不能干活**：``enabled`` 且生效模式
        （``VideoConfig.effective_provider_mode``）的依赖满足。判据只有一处定义
        （模块级纯函数 :func:`mode_is_usable`），这里调用它，不复制逻辑。

        与旧实现的区别：旧版本拿"降级"当"未就绪"，于是
        ``provider_mode=openai_compat`` 但三要素没配齐的用户（插件已自动降级成
        ``gemini_native``、视频理解实际正常工作）在宿主清单里被标成"尚未就绪"。
        现在按能力报：**降级但可用仍算就绪**，只有总开关关掉、或生效模式的依赖
        真不满足时才是 ``False``。

        两个 ``bool`` 都显式 ``bool(...)``：宿主的 ``type() is bool`` 不认
        ``1`` / ``0`` / ``"true"`` / 任何 numpy 标量。

        Returns:
            ``{"schema", "plugin_id", "enabled", "runtime_ready"}``（恰好四个
            键）。``runtime_ready`` 是 ``bool``，不会是 ``None`` —— 本插件能自己
            判断就绪性，把"未知"留给真正不知道的提供方。

        Note:
            绝不返回配置原文、API key、缓存路径或错误文本（模块说明的
            ``:208-209``）。
        """
        config = getattr(self._plugin, "config", None)
        enabled = bool(getattr(config, "enable", False))
        # 取不到 config（贴错了对象 / 老版本还没有这个字段）时按"未就绪"报，
        # 而不是抛异常让宿主显示 degraded：状态接口自己坏了才该是 degraded。
        runtime_ready = bool(config is not None and mode_is_usable(config))
        return {
            "schema": STATUS_SCHEMA,
            "plugin_id": PLUGIN_ID,
            "enabled": enabled,
            "runtime_ready": runtime_ready,
        }
