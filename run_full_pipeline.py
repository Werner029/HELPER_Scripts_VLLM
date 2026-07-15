import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

POSTS_DATA = os.path.join(DATA_DIR, "posts_universal.json")
DIALOGS_DATA = os.path.join(DATA_DIR, "dialogs_anonymous.json")
REVIEWS_DATA = os.path.join(DATA_DIR, "reviews_universal.json")

PORT = 8000


@dataclass
class ServerConfig:
    name: str
    model: str
    port: int = PORT
    gpus: int = 1
    cuda_devices: str = "0,1,2,3,4,5,6,7"
    lora_name: Optional[str] = None
    lora_path: Optional[str] = None
    max_lora_rank: int = 64


@dataclass
class ScriptTask:
    script: str
    data_file: str
    input_arg: str = "--input"


SERVERS = {
    "9b_35": ServerConfig(
        name="9b_35",
        model="Qwen/Qwen3.5-9B",
        gpus=8,
        cuda_devices="0,1,2,3,4,5,6,7",
    ),
}

TASKS: list[ScriptTask] = [
    ScriptTask("script_segm_posts.py", POSTS_DATA, "--input"),
    ScriptTask("script_tags_posts.py", POSTS_DATA, "--input"),
    ScriptTask("script_tone_posts.py", POSTS_DATA, "--input"),
    ScriptTask("script_tone_comments.py", POSTS_DATA, "--input"),
    ScriptTask("script_segm_dialogs.py", DIALOGS_DATA, "--files"),
    ScriptTask("script_tags_dialogs.py", DIALOGS_DATA, "--files"),
    ScriptTask("script_psych_dialogs.py", DIALOGS_DATA, "--files"),
    ScriptTask("script_questions_dialogs.py", DIALOGS_DATA, "--files"),
    ScriptTask("script_segm_reviews_without_markup.py", REVIEWS_DATA, "--input"),
    ScriptTask("script_tags_reviews_without_markup.py", REVIEWS_DATA, "--input"),
    ScriptTask("script_tone_reviews_without_markup.py", REVIEWS_DATA, "--input"),
]


def log(tag: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def start_server(cfg: ServerConfig, max_model_len: int = 32768) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.model,
        "--port", str(cfg.port),
        "--tensor-parallel-size", str(cfg.gpus),
        "--gpu-memory-utilization", "0.90",
        "--dtype", "auto",
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
    ]

    served_name = os.path.basename(cfg.model.rstrip("/"))
    cmd += ["--served-model-name", served_name]

    if cfg.lora_path:
        cmd += ["--enable-lora"]
        cmd += ["--lora-modules", f"{cfg.lora_name}={cfg.lora_path}"]
        cmd += ["--max-lora-rank", str(cfg.max_lora_rank)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    env["VLLM_USE_V1"] = "0"
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

    log_path = os.path.join(BASE_DIR, f"vllm_{cfg.name}.log")
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    log(cfg.name, f"SERVER START  model={cfg.model}  port={cfg.port}  TP={cfg.gpus}  GPU={cfg.cuda_devices}")
    log(cfg.name, f"Логи: {log_path}")
    return proc


def stop_server(cfg: ServerConfig, proc: subprocess.Popen):
    log(cfg.name, "SERVER STOP")
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def wait_for_server(port: int, tag: str, timeout: int = 600) -> bool:
    import urllib.request
    import urllib.error
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    return False


def _ids_from_post_link(link: str) -> str:
    m = re.search(r"wall-\d+_(\d+)", link or "")
    return m.group(1) if m else ""


def _get_expected_post_ids(data_path: str) -> set[str]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = set()
    for item in data:
        if item.get("type") != "post":
            continue
        link = item.get("source", {}).get("link", "")
        pid = _ids_from_post_link(link)
        if pid:
            ids.add(pid)
    return ids


def _get_done_post_ids(out_dir: str) -> set[str]:
    done = set()
    if not os.path.isdir(out_dir):
        return done
    for name in os.listdir(out_dir):
        if not name.startswith("group_") or not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(out_dir, name), "r", encoding="utf-8") as f:
                group = json.load(f)
            for post in group.get("posts", []):
                pid = post.get("post_id", "")
                if pid:
                    done.add(pid)
        except (json.JSONDecodeError, OSError):
            continue
    return done


def _get_expected_dialog_ids(data_path: str) -> set[str]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = set()
    for item in data.get("$values", []):
        link = item.get("Link", "")
        if "id=" in link:
            sid = link.split("id=", 1)[1].split("&", 1)[0]
            if sid:
                ids.add(sid)
    return ids


def _get_done_dialog_ids(out_dir: str) -> set[str]:
    done = set()
    if not os.path.isdir(out_dir):
        return done
    for name in os.listdir(out_dir):
        m = re.match(r"^result_(.+?)_", name)
        if m:
            done.add(m.group(1))
    return done


def _get_expected_review_count(data_path: str) -> int:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)


def _get_done_review_ids(out_dir: str) -> set[str]:
    done = set()
    if not os.path.isdir(out_dir):
        return done
    for name in os.listdir(out_dir):
        m = re.match(r"^result_(\d+)", name)
        if m:
            done.add(m.group(1))
    return done


def task_is_complete(task: ScriptTask, out_dir: str) -> tuple[bool, str]:
    if not os.path.isdir(out_dir):
        return False, "нет папки"

    script = task.script

    if script in (
        "script_segm_posts.py", "script_tags_posts.py",
        "script_tone_posts.py", "script_tone_comments.py",
    ):
        expected = _get_expected_post_ids(task.data_file)
        done = _get_done_post_ids(out_dir)
        missing = expected - done
        status = f"постов {len(done)}/{len(expected)}"
        if missing:
            status += f", не хватает {len(missing)}"
        return not missing, status

    if script in (
        "script_segm_dialogs.py", "script_tags_dialogs.py",
        "script_psych_dialogs.py", "script_questions_dialogs.py",
    ):
        expected = _get_expected_dialog_ids(task.data_file)
        done = _get_done_dialog_ids(out_dir)
        missing = expected - done
        status = f"диалогов {len(done)}/{len(expected)}"
        if missing:
            status += f", не хватает {len(missing)}"
        return not missing, status

    if script in (
        "script_segm_reviews_without_markup.py",
        "script_tags_reviews_without_markup.py",
        "script_tone_reviews_without_markup.py",
    ):
        expected_n = _get_expected_review_count(task.data_file)
        expected = {str(i) for i in range(expected_n)}
        done = _get_done_review_ids(out_dir)
        missing = expected - done
        status = f"отзывов {len(done)}/{len(expected)}"
        if missing:
            status += f", не хватает {len(missing)}"
        return not missing, status

    has_json = any(f.endswith((".json", ".jsonl")) for f in os.listdir(out_dir))
    return has_json, "fallback check"


def main():
    p = argparse.ArgumentParser(description="Пайплайн: модели последовательно, 1 сервер, 11 скриптов")
    p.add_argument("--configs", nargs="+", default=None,
                   choices=list(SERVERS.keys()),
                   help="Какие конфигурации запускать (default: все)")
    p.add_argument("--scripts", nargs="+", default=None,
                   help="Фильтр по именам скриптов (без .py)")
    p.add_argument("--lora-8b-path", type=str, default=None)
    p.add_argument("--lora-14b-path", type=str, default=None)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--skip-servers", action="store_true",
                   help="Не запускать/останавливать серверы (уже запущен на порту 8000)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Пропускать задачи, где все ID уже обработаны")
    p.add_argument("--out-base", type=str, default="results")
    p.add_argument("--server-timeout", type=int, default=600)
    args = p.parse_args()

    if args.lora_8b_path:
        SERVERS["8b_lora"].lora_path = args.lora_8b_path
    if args.lora_14b_path:
        SERVERS["14b_lora"].lora_path = args.lora_14b_path

    config_names = args.configs or list(SERVERS.keys())

    for cn in list(config_names):
        cfg = SERVERS[cn]
        if cfg.lora_name and not cfg.lora_path:
            log(cn, "LoRA путь не указан! Пропускаем. Используй --lora-8b-path / --lora-14b-path")
            config_names.remove(cn)

    tasks = TASKS
    if args.scripts:
        tasks = [t for t in TASKS if any(s in t.script for s in args.scripts)]

    out_base = os.path.join(BASE_DIR, args.out_base)
    os.makedirs(out_base, exist_ok=True)

    total = len(config_names) * len(tasks)
    done = 0
    failed = 0
    skipped = 0

    print("\n" + "=" * 60)
    print(f"ПАЙПЛАЙН: {len(config_names)} моделей × {len(tasks)} скриптов = {total} задач")
    print(f"Режим: последовательный (1 сервер, модели по очереди)")
    if args.skip_existing:
        print("Пропуск готовых: ВКЛ")
    print("=" * 60)

    for cn in config_names:
        cfg = SERVERS[cn]

        if not args.skip_servers:
            proc = start_server(cfg, args.max_model_len)
            log(cn, "Ожидание готовности сервера...")
            if not wait_for_server(cfg.port, cn, args.server_timeout):
                log(cn, "СЕРВЕР НЕ ПОДНЯЛСЯ! Пропускаем конфигурацию.")
                for t in tasks:
                    done += 1
                    failed += 1
                stop_server(cfg, proc)
                continue
            log(cn, "СЕРВЕР ГОТОВ")
        else:
            proc = None

        for task in tasks:
            task_out_dir = os.path.join(out_base, cfg.name, os.path.splitext(task.script)[0])
            script_path = os.path.join(BASE_DIR, task.script)

            if args.skip_existing:
                complete, status = task_is_complete(task, task_out_dir)
                if complete:
                    log(cn, f"SKIP  {task.script}  ({status})")
                    done += 1
                    skipped += 1
                    continue

            if not os.path.exists(script_path):
                log(cn, f"FAIL  {task.script}  скрипт не найден: {script_path}")
                done += 1
                failed += 1
                continue

            if not os.path.exists(task.data_file):
                log(cn, f"FAIL  {task.script}  данные не найдены: {task.data_file}")
                done += 1
                failed += 1
                continue

            served_name = os.path.basename(cfg.model.rstrip("/"))
            cmd = [
                sys.executable, script_path,
                "--api", f"http://localhost:{cfg.port}",
                "--model", served_name,
                task.input_arg, task.data_file,
                "--out", task_out_dir,
                "--max-tokens", "4096",
            ]

            task_cwd = os.path.join(out_base, cfg.name)
            os.makedirs(task_cwd, exist_ok=True)

            t0 = time.time()
            log(cn, f"START  {task.script}")

            p = subprocess.Popen(cmd, cwd=task_cwd)
            p.wait()
            elapsed = time.time() - t0
            done += 1

            if p.returncode != 0:
                log(cn, f"FAIL  {task.script}  ({fmt_duration(elapsed)})  exit={p.returncode}")
                failed += 1
            else:
                log(cn, f"DONE   {task.script}  ({fmt_duration(elapsed)})")

        if proc is not None:
            stop_server(cfg, proc)

    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)
    print(f"  Всего:     {done}/{total}")
    print(f"  Успешно:   {done - failed - skipped}")
    print(f"  Ошибок:    {failed}")
    print(f"  Пропущено: {skipped}")
    print(f"  Результаты: {out_base}/")
    print()
    print("Структура результатов:")
    print(f"  {out_base}/")
    for cn in config_names:
        print(f"    {cn}/")
        for task in tasks:
            print(f"      {os.path.splitext(task.script)[0]}/")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
