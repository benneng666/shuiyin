from __future__ import annotations

import argparse
import asyncio
import json
import queue
import signal
import threading
from dataclasses import asdict
from pathlib import Path

from .links import extract_links


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="商品素材采集测试版：主图、详情图、商品视频及评价/追评图片视频，保留素材来源。",
        epilog="三平台适配待真实页面验收；小红书商品评价尚未支持。仅接收商品链接，不采集笔记或短视频作品评论。")
    parser.add_argument("links", nargs="*", help="商品链接或分享文案（含空格时加引号）")
    parser.add_argument("--file", type=Path, help="UTF-8 文本，每行一个商品链接或分享文案")
    parser.add_argument("--check", action="store_true", help="仅检查链接，不打开浏览器、不联网下载")
    parser.add_argument("--gui", action="store_true", help="打开 Windows 窗口")
    parser.add_argument("--output", type=Path, default=Path("downloads"), help="图片、视频和来源清单的保存目录")
    parser.add_argument("--profiles", type=Path, default=Path(".profiles"), help="软件独立浏览器会话目录；不共享日常浏览器登录")
    parser.add_argument("--max-scrolls", type=int, default=40)
    parser.add_argument("--channel", choices=["msedge", "chrome", "chromium"], default="msedge")
    args = parser.parse_args(argv)
    if args.gui:
        from .gui import main as gui_main
        gui_main()
        return 0
    text = "\n".join(args.links)
    if args.file:
        try:
            text += "\n" + args.file.read_text(encoding="utf-8-sig")
        except OSError:
            parser.error("无法读取链接文件")
    try:
        links = extract_links(text)
    except ValueError as exc:
        parser.error(str(exc))
    if args.check:
        print(json.dumps([asdict(x) for x in links], ensure_ascii=False, indent=2))
        return 0
    if not 1 <= args.max_scrolls <= 1000:
        parser.error("--max-scrolls 须为 1 至 1000")

    stopped = threading.Event()

    async def ask(message):
        answers = queue.Queue()
        print(message + "\n请在软件打开的独立 Edge / Chrome 中登录或验证；日常浏览器登录不共享。"
              "\n处理完成输入 y 继续，其他输入跳过：", flush=True)

        def read_answer():
            try:
                answers.put(input())
            except (EOFError, OSError, RuntimeError):
                answers.put("")

        # A cancelled console prompt must not hold asyncio's executor open.
        threading.Thread(target=read_answer, daemon=True).start()
        while not stopped.is_set():
            try:
                return answers.get_nowait().strip().lower() == "y"
            except queue.Empty:
                await asyncio.sleep(0.1)
        return False

    def stop_signal(signum, frame):
        if stopped.is_set():
            raise KeyboardInterrupt
        stopped.set()
        print("正在停止并保存已读取记录；再次 Ctrl+C 将强制退出。", flush=True)

    previous_signal = signal.signal(signal.SIGINT, stop_signal)
    try:
        from .engine import run_batch
        reports = asyncio.run(run_batch(links, args.output, args.profiles, print, ask,
                                       args.max_scrolls, args.channel, should_stop=stopped.is_set))
    except ImportError:
        print("缺少运行依赖。请先运行安装脚本或 pip install -r requirements.txt。")
        return 2
    except KeyboardInterrupt:
        print("已停止。已写入磁盘的图片和记录保留。")
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_signal)
    totals = {"unique_images": 0, "unique_videos": 0, "downloads_failed": 0, "videos_failed": 0}
    for report in reports:
        stats = report.get("stats") if isinstance(report, dict) else None
        if not isinstance(stats, dict):
            continue
        for key in totals:
            value = stats.get(key, 0)
            if type(value) is int and value >= 0:
                totals[key] += value
    print(f"已保存（按任务合计）：图片 {totals['unique_images']}，视频 {totals['unique_videos']}；"
          f"下载失败 {totals['downloads_failed'] + totals['videos_failed']}。素材来源见商品清单。")
    if stopped.is_set():
        return 130
    return 0 if reports and all(r.get("complete") for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
