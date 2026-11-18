"""Windows desktop test build; importing this module starts no UI or browser."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import queue
import re
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from urllib.parse import urlsplit, urlunsplit
from buyer_show import __version__


APP_TITLE = f"买家秀采集助手 {__version__} · 测试版"
LOGGER = logging.getLogger("buyer_show")
BROWSERS = {"Edge": "msedge", "Chrome": "chrome"}
PLATFORMS = {"taobao": "淘宝 / 天猫", "douyin": "抖音", "xiaohongshu": "小红书"}
STATUSES = {
    "queued": "等待开始", "running": "采集中", "waiting_user": "等待人工处理",
    "completed": "已完成", "partial": "未完成", "partial_download": "下载未完成",
    "login_required": "需要登录", "verification_required": "需要人工验证",
    "access_restricted": "访问受限", "unavailable": "商品不可用",
    "unsupported": "尚未支持", "unsupported_link": "链接不支持", "error": "执行失败",
    "cancelled": "已停止", "not_started": "未开始", "stopped_pending": "未开始（已停止）",
    "unsupported_review_entry": "评价入口未支持", "unsupported_page": "页面尚未适配",
    "app_required": "需要App查看", "product_unavailable": "商品不可用",
}
STAGES = {"opening_browser": "打开浏览器", "read_reviews": "读取评价",
          "download_reviews": "下载买家秀", "read_product": "读取商品素材",
          "download_product": "下载商品素材", "organize_categories": "整理分类"}


def _task_status(row):
    if row.get("status") == "running" and row.get("stage") in STAGES:
        return STAGES[row["stage"]]
    return STATUSES.get(row.get("status"), "未完成")


def _discovered_text(stats):
    if not isinstance(stats, dict) or not any(key in stats for key in ("image_urls", "video_urls")):
        return "发现链接数量尚未记录"
    def value(key):
        number = stats.get(key)
        return str(number) if type(number) is int and number >= 0 else "未记录"
    return f"已发现链接：图片 {value('image_urls')}，视频 {value('video_urls')}（不是保存数量）"


def _media_counts(stats):
    """Accept current and image-only manifests without inventing video counts."""
    stats = stats if isinstance(stats, dict) else {}
    def count(key):
        value = stats.get(key, 0)
        return value if type(value) is int and value >= 0 else 0
    return {
        "images": count("unique_images"), "videos": count("unique_videos"),
        "failed": count("downloads_failed") + count("videos_failed"),
        "reviews": count("reviews_read"), "product_media": count("product_media_count"),
    }


def _media_totals(rows):
    totals = {"images": 0, "videos": 0, "failed": 0}
    for row in rows:
        if isinstance(row, dict):
            counts = _media_counts(row.get("stats"))
            for key in totals:
                totals[key] += counts[key]
    return totals


def _record_exception(error, context, *, tb=None, level=logging.WARNING):
    """Record types and frame positions only, never exception text or locals."""
    error_type = error if isinstance(error, type) else type(error)
    name = error_type.__name__
    if tb is None and not isinstance(error, type):
        tb = error.__traceback__
    frames = [f"{frame.filename}:{frame.lineno} in {frame.name}"
              for frame in traceback.extract_tb(tb)] if tb else []
    LOGGER.log(level, "%s：%s%s", context, name,
               "\n" + "\n".join(frames) if frames else "")
    return name


def _safe_message(value: object) -> str:
    """Keep progress useful without echoing URL queries or credentials."""
    text = str(value)
    text = re.sub(r"(?im)(?:set-cookie|cookie|authorization)\s*[:=][^\r\n]*",
                  "[认证信息已隐藏]", text)
    text = re.sub(
        r"(?i)([\"']?(?:access_token|refresh_token|xsec_token|token|password|secret|sessionid|sign|x-s)[\"']?\s*[:=]\s*)[^\s,;]+",
        r"\1[已隐藏]", text,
    )

    def safe_url(match):
        try:
            url = urlsplit(match.group(0))
            return urlunsplit((url.scheme, url.hostname or "", url.path, "", ""))
        except ValueError as exc:
            _record_exception(exc, "日志链接脱敏", level=logging.DEBUG)
            return "[链接已隐藏]"

    text = re.sub(r"https?://[^\s<>\"']+", safe_url, text)
    return text[:4000] + ("…" if len(text) > 4000 else "")


def _normalized_settings(values=None):
    values = values if isinstance(values, dict) else {}
    output = values.get("output")
    if not isinstance(output, str) or not output.strip() or "\x00" in output:
        output = str(Path.home() / "Downloads" / "商品素材")
    try:
        scrolls = int(values.get("max_scrolls", 200))
        if isinstance(values.get("max_scrolls"), bool) or not 1 <= scrolls <= 1000:
            scrolls = 200
    except (ValueError, TypeError, OverflowError) as exc:
        _record_exception(exc, "加载上限配置回退", level=logging.INFO)
        scrolls = 200
    channel = values.get("channel")
    if channel not in BROWSERS.values():
        channel = "msedge"
    return {"output": output.strip(), "max_scrolls": scrolls, "channel": channel}


def _load_startup():
    """A missing/corrupt settings file must not prevent opening the app."""
    values = {}
    profiles = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "BuyerShowAssistant" / "profiles"
    warning = False
    try:
        from buyer_show.runtime import app_data_dir, load_settings
    except Exception as exc:
        _record_exception(exc, "配置模块加载失败")
        return _normalized_settings(), profiles, True
    try:
        profiles = Path(app_data_dir()) / "profiles"
    except Exception as exc:
        _record_exception(exc, "会话目录配置回退")
        warning = True
    try:
        values = load_settings()
        if not isinstance(values, dict):
            warning = True
    except Exception as exc:
        _record_exception(exc, "读取配置失败")
        warning = True
    return _normalized_settings(values), profiles, warning


def _read_link_text(path):
    with Path(path).open("rb") as stream:
        data = stream.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("TXT 文件过大，请拆分为不超过 2 MB 的文件。")
    encodings = ("utf-16",) if data.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            if "\x00" not in text:
                return text
        except UnicodeError as exc:
            _record_exception(exc, "TXT 编码尝试", level=logging.DEBUG)
            continue
    raise ValueError("无法读取文本编码，请将文件另存为 UTF-8 格式后重试。")


class _Prompt:
    def __init__(self, message: str):
        self.message = message
        self.event = threading.Event()
        self.approved = False


class ValidatorWindow:
    def __init__(self, root: tk.Tk, settings=None, profiles=None, settings_warning=False):
        settings = _normalized_settings(settings)
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.pending_lock = threading.Lock()
        self.pending: _Prompt | None = None
        self.worker: threading.Thread | None = None
        self.running = False
        self.close_requested = False
        self.done_received = False
        self.last_summary: dict = {}
        self.task_rows: dict[int, dict] = {}
        self.current_output: Path | None = None
        self.settings_save_warning = False
        self.watermark_window = None
        self.profiles = Path(profiles) if profiles is not None else _load_startup()[1]
        self.output = tk.StringVar(value=settings["output"])
        self.max_scrolls = tk.StringVar(value=str(settings["max_scrolls"]))
        self.browser = tk.StringVar(value=next(name for name, channel in BROWSERS.items() if channel == settings["channel"]))
        self.status = tk.StringVar(value="就绪。粘贴商品链接或导入 TXT 后开始。")
        self.prompt = tk.StringVar(value="需要登录或验证时，请在软件打开的独立 Edge / Chrome 中处理，再点“处理完成，继续”；不共享日常浏览器登录。")
        self.task_detail = tk.StringVar(value="选择任务查看说明；双击结果行打开该商品的素材分类文件夹。")
        self.root.report_callback_exception = self._callback_exception
        self._build()
        if settings_warning:
            self._append_log("部分配置未能读取，已使用默认设置；可重新选择下载目录和浏览器。")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll)

    def _build(self):
        self.root.title(APP_TITLE)
        self.root.geometry("960x760")
        self.root.minsize(760, 650)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame = ttk.Frame(self.root, padding=14)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(6, weight=3)
        ttk.Label(frame, text=APP_TITLE, font=("Microsoft YaHei UI", 17, "bold")).grid(row=0, sticky="w")
        notice = ttk.Label(frame,
            text="采集范围：当前商品主图、详情图、商品视频，以及评价 / 追评买家秀图片与视频；保留来源，推荐区不采集。\n"
                 "三平台适配仍待真实页面验收；小红书商品评价尚未支持。停滞、中断或达到加载上限仍标记未完成。",
            wraplength=900, justify="left")
        notice.grid(row=1, sticky="ew", pady=(6, 10))
        input_frame = ttk.LabelFrame(frame, text="商品链接 / 分享文案（支持多行）", padding=8)
        input_frame.grid(row=2, sticky="nsew")
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)
        ttk.Label(input_frame, text="仅接收商品链接；不接收笔记、短视频作品或店铺主页链接。").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.import_button = ttk.Button(input_frame, text="导入 TXT…", command=self._import_text)
        self.import_button.grid(row=0, column=1, sticky="e", pady=(0, 5))
        self.input = ScrolledText(input_frame, height=4, wrap="word", font=("Microsoft YaHei UI", 10))
        self.input.grid(row=1, column=0, columnspan=2, sticky="nsew")

        options = ttk.Frame(frame)
        options.grid(row=3, sticky="ew", pady=(10, 8))
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="下载目录").grid(row=0, column=0, padx=(0, 8))
        self.output_entry = ttk.Entry(options, textvariable=self.output)
        self.output_entry.grid(row=0, column=1, sticky="ew")
        self.output_entry.bind("<FocusOut>", lambda event: self._remember_settings())
        self.browse_button = ttk.Button(options, text="选择…", command=self._browse)
        self.browse_button.grid(row=0, column=2, padx=6)
        ttk.Button(options, text="打开目录", command=self._open_output).grid(row=0, column=3)
        preferences = ttk.Frame(options)
        preferences.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(preferences, text="浏览器").pack(side="left", padx=(0, 8))
        self.browser_combo = ttk.Combobox(preferences, textvariable=self.browser, values=list(BROWSERS), state="readonly", width=10)
        self.browser_combo.pack(side="left")
        self.browser_combo.bind("<<ComboboxSelected>>", lambda event: self._remember_settings())
        ttk.Label(preferences, text="每个商品加载上限").pack(side="left", padx=(18, 8))
        self.max_scrolls_spinbox = ttk.Spinbox(preferences, from_=1, to=1000, increment=1, width=6,
            textvariable=self.max_scrolls, command=self._remember_settings)
        self.max_scrolls_spinbox.pack(side="left")
        self.max_scrolls_spinbox.bind("<FocusOut>", lambda event: self._remember_settings())
        ttk.Label(preferences, text="次（1–1000；默认 200）").pack(side="left", padx=(5, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=4, sticky="ew")
        self.start_button = ttk.Button(actions, text="开始采集", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.continue_button = ttk.Button(actions, text="处理完成，继续", command=self._continue, state="disabled")
        self.continue_button.pack(side="left")
        self.watermark_button = ttk.Button(actions, text="去水印…", command=self._open_watermark)
        self.watermark_button.pack(side="right")
        prompt_label = ttk.Label(frame, textvariable=self.prompt, wraplength=900, justify="left")
        prompt_label.grid(row=5, sticky="ew", pady=(7, 9))

        self.notebook = ttk.Notebook(frame)
        self.notebook.grid(row=6, sticky="nsew")
        self.tasks_tab = ttk.Frame(self.notebook, padding=7)
        log_tab = ttk.Frame(self.notebook, padding=7)
        self.notebook.add(self.tasks_tab, text="任务")
        self.notebook.add(log_tab, text="日志")
        self.tasks_tab.columnconfigure(0, weight=1)
        self.tasks_tab.rowconfigure(0, weight=1)
        columns = ("platform", "product", "status", "saved", "videos", "failed")
        self.tasks = ttk.Treeview(self.tasks_tab, columns=columns, show="headings", selectmode="browse", height=6)
        for column, title, width, minimum in (
            ("platform", "平台", 110, 90), ("product", "商品 ID / 短链", 220, 160),
            ("status", "阶段 / 状态", 150, 120), ("saved", "已存图片", 85, 75),
            ("videos", "已存视频", 85, 75), ("failed", "下载失败", 80, 70),
        ):
            self.tasks.heading(column, text=title)
            self.tasks.column(column, width=width, minwidth=minimum,
                              anchor="center" if column in {"saved", "videos", "failed"} else "w", stretch=column in {"product", "status"})
        self.tasks.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self.tasks_tab, orient="vertical", command=self.tasks.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(self.tasks_tab, orient="horizontal", command=self.tasks.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.tasks.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
        self.tasks.bind("<<TreeviewSelect>>", self._select_task)
        self.tasks.bind("<Double-1>", self._open_task)
        detail_label = ttk.Label(self.tasks_tab, textvariable=self.task_detail, wraplength=870, justify="left")
        detail_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(0, weight=1)
        self.log = ScrolledText(log_tab, height=8, state="disabled", wrap="word", font=("Microsoft YaHei UI", 9))
        self.log.grid(sticky="nsew")
        status_label = ttk.Label(frame, textvariable=self.status, wraplength=900, justify="left")
        status_label.grid(row=7, sticky="ew", pady=(8, 0))
        # Rewrap explanatory text instead of forcing a wide minimum window.
        def rewrap(event):
            for label in (notice, prompt_label, status_label):
                label.configure(wraplength=max(300, event.width - 6))
            detail_label.configure(wraplength=max(300, event.width - 34))
        frame.bind("<Configure>", rewrap)

    def _append_log(self, message):
        safe_message = _safe_message(message)
        LOGGER.info("%s", safe_message)
        self.log.configure(state="normal")
        self.log.insert("end", safe_message + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 5000:
            self.log.delete("1.0", "1000.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _callback_exception(self, error_type, _error_value, tb):
        name = _record_exception(error_type, "界面回调失败", tb=tb, level=logging.ERROR)
        try:
            messagebox.showerror("界面操作未完成",
                f"界面操作出现错误（{name}），已记录到本地日志。\n"
                "如任务仍在运行，可点击停止，待任务结束后重新尝试。", parent=self.root)
        except Exception as exc:
            _record_exception(exc, "显示界面错误提示失败", level=logging.ERROR)

    def _remember_settings(self):
        try:
            scrolls = int(self.max_scrolls.get())
            if not 1 <= scrolls <= 1000 or not self.output.get().strip():
                return
            values = {"output": self.output.get().strip(), "max_scrolls": scrolls, "channel": BROWSERS[self.browser.get()]}
        except (ValueError, KeyError) as exc:
            _record_exception(exc, "设置尚未保存", level=logging.INFO)
            return
        try:
            from buyer_show.runtime import save_settings
            save_settings(values)
        except Exception as exc:
            _record_exception(exc, "保存配置失败")
            if not self.settings_save_warning:
                self.settings_save_warning = True
                self._append_log("设置暂未保存；本次仍使用界面中的选项，下次启动可能需要重新选择。")

    def _browse(self):
        directory = filedialog.askdirectory(parent=self.root, title="选择下载目录")
        if directory:
            self.output.set(directory)
            self._remember_settings()

    def _import_text(self):
        if self.running:
            return
        path = filedialog.askopenfilename(parent=self.root, title="导入商品链接 TXT", filetypes=[("文本文件", "*.txt")])
        if not path:
            return
        try:
            text = _read_link_text(path).strip()
        except ValueError as exc:
            _record_exception(exc, "TXT 导入校验失败", level=logging.INFO)
            messagebox.showerror("无法导入 TXT", str(exc), parent=self.root)
            return
        except OSError as exc:
            _record_exception(exc, "TXT 读取失败")
            messagebox.showerror("无法导入 TXT", "文件无法读取，请检查路径和访问权限。", parent=self.root)
            return
        if text:
            if self.input.get("1.0", "end").strip():
                self.input.insert("end", "\n")
            self.input.insert("end", text + "\n")
            self._append_log("已导入 TXT 内容，请核对商品链接后开始。")
        else:
            messagebox.showinfo("TXT 内容为空", "文件中没有可导入的文本。", parent=self.root)

    def _initialize_tasks(self, links):
        children = self.tasks.get_children()
        if children:
            self.tasks.delete(*children)
        self.task_rows = {}
        for index, link in enumerate(links):
            self.task_rows[index] = {"platform": link.platform, "product_id": link.product_id,
                                     "status": "queued", "reason": "等待开始。", "stats": {}}
            self.tasks.insert("", "end", iid=str(index), values=self._row_values(self.task_rows[index]))
        self.task_detail.set("选择任务查看说明；双击结果行打开该商品的素材分类文件夹。")
        self.notebook.select(self.tasks_tab)

    @staticmethod
    def _row_values(row):
        counts = _media_counts(row.get("stats"))
        return (PLATFORMS.get(row.get("platform"), "未知平台"), row.get("product_id") or "待解析短链",
                _task_status(row), counts["images"], counts["videos"], counts["failed"])

    def _task_event(self, event):
        if not isinstance(event, dict) or event.get("type") != "task":
            return
        index = event.get("index")
        if type(index) is not int or index not in self.task_rows:
            return
        row = self.task_rows[index]
        for key in ("platform", "product_id", "status", "manifest", "stage"):
            if event.get(key) is not None:
                row[key] = str(event[key])
        if "reason" in event:
            row["reason"] = _safe_message(event["reason"])
        if "download_failure_summary" in event:
            row["download_failure_summary"] = _safe_message(event["download_failure_summary"] or "")
        if isinstance(event.get("stats"), dict):
            row["stats"] = dict(event["stats"])
        self.tasks.item(str(index), values=self._row_values(row))
        if event.get("status") in {"running", "waiting_user"}:
            self.tasks.see(str(index))
        if event.get("status") == "running" and event.get("stage") in STAGES:
            counts = _media_counts(row.get("stats"))
            stop_event = getattr(self, "stop_event", None)
            if not stop_event or not stop_event.is_set():
                self.status.set(f"{_task_status(row)} · {_discovered_text(row.get('stats'))}；"
                                f"已保存：图片 {counts['images']}，视频 {counts['videos']}；"
                                f"已读取评价 {counts['reviews']}。")
        self._select_task()

    def _select_task(self, event=None):
        selection = self.tasks.selection()
        if not selection:
            return
        row = self.task_rows.get(int(selection[0]))
        if row:
            reason = row.get("reason") or STATUSES.get(row.get("status"), "未完成")
            counts = _media_counts(row.get("stats"))
            failures = _safe_message(row.get("download_failure_summary") or "")[:240]
            self.task_detail.set(
                f"{_task_status(row)} · {_discovered_text(row.get('stats'))}。\n"
                f"已保存：图片 {counts['images']}，视频 {counts['videos']}；下载失败 {counts['failed']}。"
                f"已读取评价 {counts['reviews']}，商品素材记录 {counts['product_media']}；来源见清单。\n"
                + (failures + "\n" if failures else "")
                + _safe_message(reason)[:220])

    def _finish_pending_rows(self):
        stopped = self.stop_event.is_set()
        for index, row in self.task_rows.items():
            if row["status"] == "queued":
                self._task_event({"type": "task", "index": index,
                    "status": "stopped_pending" if stopped else "not_started", "reason": "本次尚未开始该商品的采集。"})
            elif row["status"] in {"running", "waiting_user"}:
                self._task_event({"type": "task", "index": index,
                    "status": "cancelled" if stopped else "partial", "reason": "任务中断，完整性尚未确认；已有结果以商品文件夹为准。"})

    def _start(self):
        watermark = getattr(self, "watermark_window", None)
        if watermark is not None and not watermark.closed:
            messagebox.showinfo("请先关闭去水印窗口", "请先完成或取消去水印处理并关闭该窗口，再开始采集，避免争用本机资源。", parent=self.root)
            watermark.focus()
            return
        if self.running or self.close_requested:
            return
        text = self.input.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("尚未输入商品链接", "请粘贴商品链接、商品分享文案，或导入 TXT。", parent=self.root)
            return
        try:
            from buyer_show.links import extract_links
            links = extract_links(text)
        except ValueError as exc:
            _record_exception(exc, "商品链接校验失败", level=logging.INFO)
            messagebox.showerror("商品链接不适用", str(exc), parent=self.root)
            return
        except Exception as exc:
            _record_exception(exc, "商品链接解析失败", level=logging.ERROR)
            messagebox.showerror("无法解析链接", "请检查商品链接格式，并确认程序已完整安装。", parent=self.root)
            return
        if not links:
            messagebox.showinfo("没有识别到商品链接", "仅支持商品链接，不处理笔记、短视频作品或店铺主页链接。", parent=self.root)
            return
        try:
            max_scrolls = int(self.max_scrolls.get())
            if not 1 <= max_scrolls <= 1000:
                raise ValueError("out of range")
        except ValueError as exc:
            _record_exception(exc, "加载上限校验失败", level=logging.INFO)
            messagebox.showerror("加载上限无效", "请输入 1 到 1000 之间的整数。达到上限后仍会标记未完成。", parent=self.root)
            return
        channel = BROWSERS.get(self.browser.get())
        if channel is None:
            messagebox.showerror("请选择浏览器", "请选择已安装的 Edge 或 Chrome。", parent=self.root)
            return
        try:
            directory = self.output.get().strip()
            if not directory:
                raise ValueError("empty")
            output = Path(directory).expanduser().resolve()
            if output.exists() and not output.is_dir():
                raise ValueError("not a directory")
        except (ValueError, OSError) as exc:
            _record_exception(exc, "下载目录校验失败")
            messagebox.showerror("下载目录无效", "请选择有效的文件夹路径。", parent=self.root)
            return
        self.output.set(str(output))
        self._remember_settings()
        self.current_output = output
        self._initialize_tasks(links)
        self.stop_event.clear()
        self.running = True
        self.done_received = False
        self.last_summary = {}
        for widget in (self.input, self.output_entry, self.browse_button, self.import_button,
                       self.max_scrolls_spinbox, self.browser_combo, self.start_button):
            widget.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.continue_button.configure(state="disabled")
        self.status.set(f"正在处理 {len(links)} 个商品链接。")
        self.prompt.set(f"请在软件打开的独立 {self.browser.get()} 窗口登录或验证，再点“处理完成，继续”；日常浏览器的登录不会共享。")
        self._append_log(f"开始采集商品素材图片与视频：{len(links)} 个商品，浏览器 {self.browser.get()}，每个商品加载上限 {max_scrolls} 次；达到上限仍标记未完成。")
        self.worker = threading.Thread(target=self._run_worker, args=(links, output, max_scrolls, channel),
                                       name="buyer-show-collector", daemon=False)
        try:
            self.worker.start()
        except RuntimeError as exc:
            _record_exception(exc, "后台任务启动失败", level=logging.ERROR)
            self.running = False
            self._set_idle()
            self._finish_pending_rows()
            self.status.set("无法启动后台任务，请关闭程序后重试。")

    def _queue_task_event(self, event):
        # Executed by the worker: copy data only; Tk stays on the main thread.
        if isinstance(event, dict) and event.get("type") == "task":
            copied = {key: event[key] for key in ("type", "index", "platform", "product_id", "status", "stage",
                                                 "reason", "manifest", "download_failure_summary") if key in event}
            if isinstance(event.get("stats"), dict):
                copied["stats"] = dict(event["stats"])
            self.events.put(("task", copied))

    def _run_worker(self, links, output, max_scrolls, channel="msedge"):
        summary = {"failed": False, "jobs": 0, "completed": 0, "images": 0, "videos": 0, "downloads_failed": 0}
        async def run():
            from buyer_show.engine import run_batch
            ask_lock = asyncio.Lock()
            async def ask(message):
                async with ask_lock:
                    if self.stop_event.is_set():
                        return False
                    prompt = _Prompt(_safe_message(message))
                    with self.pending_lock:
                        self.pending = prompt
                        if self.stop_event.is_set():
                            prompt.event.set()
                    self.events.put(("ask", prompt))
                    try:
                        await asyncio.to_thread(prompt.event.wait)
                        return prompt.approved and not self.stop_event.is_set()
                    finally:
                        prompt.event.set()
                        with self.pending_lock:
                            if self.pending is prompt:
                                self.pending = None
            return await run_batch(
                links, output, self.profiles, emit=lambda message: self.events.put(("log", _safe_message(message))),
                ask=ask, max_scrolls=max_scrolls, channel=channel, should_stop=self.stop_event.is_set,
                on_event=self._queue_task_event,
            )
        try:
            results = asyncio.run(run())
            if isinstance(results, list):
                summary["jobs"] = len(results)
                summary["completed"] = sum(isinstance(row, dict) and row.get("complete") is True for row in results)
                totals = _media_totals(results)
                summary.update(images=totals["images"], videos=totals["videos"], downloads_failed=totals["failed"])
                for index, result in enumerate(results):
                    if isinstance(result, dict):
                        self._queue_task_event(dict(result, type="task", index=index))
        except (Exception, asyncio.CancelledError) as exc:
            _record_exception(exc, "后台采集未完成", level=logging.ERROR)
            summary["failed"] = True
            self.events.put(("log", "执行未完成。请检查程序依赖、所选浏览器和输出目录权限；已有结果以输出目录为准。"))
        finally:
            with self.pending_lock:
                if self.pending:
                    self.pending.event.set()
                self.pending = None
            self.events.put(("done", summary))

    def _continue(self):
        if self.stop_event.is_set() or not self.running:
            return
        with self.pending_lock:
            prompt = self.pending
            if prompt is None or prompt.event.is_set():
                return
            prompt.approved = True
            prompt.event.set()
        self.continue_button.configure(state="disabled")
        self.prompt.set("已提交继续请求，正在重新检查页面状态。")
        self.status.set("继续采集中。")

    def _stop(self):
        if not self.running:
            return
        self.stop_event.set()
        with self.pending_lock:
            if self.pending is not None:
                self.pending.approved = False
                self.pending.event.set()
        self.stop_button.configure(state="disabled")
        self.continue_button.configure(state="disabled")
        self.status.set("已请求停止，等待当前操作结束并保留结果。请稍候。")
        self.prompt.set("正在停止；不会继续自动操作或提交人工确认。")

    def _set_idle(self):
        for widget in (self.input, self.output_entry, self.browse_button, self.import_button,
                       self.max_scrolls_spinbox, self.start_button):
            widget.configure(state="normal")
        self.browser_combo.configure(state="readonly")
        self.stop_button.configure(state="disabled")
        self.continue_button.configure(state="disabled")

    def _poll(self):
        for _ in range(150):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(payload)
            elif kind == "task":
                self._task_event(payload)
            elif kind == "ask":
                if payload.event.is_set():
                    continue
                if self.stop_event.is_set() or self.close_requested:
                    payload.event.set()
                else:
                    self.prompt.set("请在软件打开的独立 Edge / Chrome 中处理，日常浏览器登录不共享。" + payload.message[:220])
                    self.continue_button.configure(state="normal")
                    self.status.set("等待人工处理。完成后点击“处理完成，继续”，或点击“停止”。")
                    self._append_log(payload.message)
            elif kind == "done":
                self.last_summary = payload
                self.done_received = True
        if self.running and self.done_received and self.worker is not None and not self.worker.is_alive():
            self.running = False
            self._set_idle()
            self._finish_pending_rows()
            counts = _media_totals(self.task_rows.values())
            media_summary = f"已保存（按任务合计）：图片 {counts['images']}，视频 {counts['videos']}；下载失败 {counts['failed']}。"
            if self.stop_event.is_set():
                self.status.set("任务已停止。" + media_summary + "请核对未完成项。")
            elif self.last_summary.get("failed"):
                self.status.set("本次未正常结束。" + media_summary + "请查看任务说明和日志。")
            else:
                completed = self.last_summary.get("completed", 0)
                self.status.set(f"本批已结束：{completed}/{len(self.task_rows)} 个商品标记完成。" + media_summary)
            self.prompt.set("人工处理等待已结束。双击任务结果可打开该商品的文件夹。")
            if self.close_requested:
                self.root.destroy()
                return
        self.root.after(100, self._poll)

    def _open_folder(self, path):
        try:
            path = Path(path).expanduser().resolve()
            if not path.is_dir():
                messagebox.showinfo("目录尚未创建", "任务保存结果后即可打开该文件夹。", parent=self.root)
                return
            os.startfile(str(path))
        except (OSError, ValueError, AttributeError) as exc:
            _record_exception(exc, "打开目录失败")
            messagebox.showerror("无法打开目录", "请通过文件资源管理器打开所选下载目录。", parent=self.root)

    def _open_output(self):
        self._open_folder(self.current_output if self.running and self.current_output else self.output.get())

    def _open_task(self, event):
        item = self.tasks.identify_row(event.y)
        if not item or self.tasks.identify_region(event.x, event.y) not in {"cell", "tree"}:
            return
        row = self.task_rows.get(int(item), {})
        try:
            if not self.current_output:
                raise ValueError("no output")
            base = self.current_output.resolve()
            if row.get("manifest"):
                folder = Path(row["manifest"]).resolve().parent
            elif row.get("platform") in PLATFORMS and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(row.get("product_id") or "")):
                folder = (base / row["platform"] / str(row["product_id"])).resolve()
            else:
                raise ValueError("unresolved")
            folder.relative_to(base)
            categorized = folder / "素材分类"
            if categorized.is_dir():
                categorized.resolve().relative_to(base)
                folder = categorized
        except (OSError, ValueError, TypeError) as exc:
            _record_exception(exc, "商品结果目录不可用", level=logging.INFO)
            messagebox.showinfo("暂无结果目录", "该任务尚未保存可打开的商品结果。", parent=self.root)
            return
        self._open_folder(folder)

    def _close(self):
        self._remember_settings()
        watermark = getattr(self, "watermark_window", None)
        if watermark is not None and not watermark.closed:
            self.root.title("正在清理去水印任务 · 结束后自动关闭")
            watermark.request_close(on_closed=self._close)
            return
        if self.running:
            self.close_requested = True
            self._stop()
            self.root.title("正在停止任务 · 结束后自动关闭")
        else:
            self.root.destroy()

    def _open_watermark(self):
        if self.running:
            messagebox.showinfo("请先停止采集", "请等待采集结束，或停止采集后再打开去水印工具。", parent=self.root)
            return
        existing = getattr(self, "watermark_window", None)
        if existing is not None and not existing.closed:
            existing.focus()
            return
        from buyer_show.watermark_ui import open_window
        directory = Path(getattr(self, "current_output", None) or self.output.get()).expanduser()
        manifests = None
        selected = self.tasks.selection()
        if selected:
            row = self.task_rows.get(int(selected[0]), {})
            manifests = []
            if row.get("manifest"):
                directory = Path(row["manifest"]).parent
                manifests = [row["manifest"]]
        elif self.task_rows:
            manifests = list(dict.fromkeys(row["manifest"] for row in self.task_rows.values() if row.get("manifest")))

        def closed():
            self.watermark_window = None

        self.watermark_window = open_window(self.root, directory, on_closed=closed, manifests=manifests, autoload=True)


def main():
    settings, profiles, warning = _load_startup()
    root = tk.Tk()
    ValidatorWindow(root, settings=settings, profiles=profiles, settings_warning=warning)
    root.mainloop()


if __name__ == "__main__":
    main()
