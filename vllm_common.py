import json
import os
import re
import sys
import time
import uuid

import aiohttp
import asyncio


async def gen(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
              api_base: str, model: str, prompt: str, max_tokens: int,
              temperature: float = 0.0, top_p: float = 0.95,
              timeout: float = 1200.0, stats_path: str | None = None,
              meta: dict | None = None) -> str:
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            j = await r.json()
    content = j["choices"][0]["message"]["content"]
    usage = j.get("usage")
    finish_reason = j["choices"][0].get("finish_reason")
    if stats_path is not None:
        rec = {"ts": time.time(), "req_id": str(uuid.uuid4()), "model": model,
               "max_tokens": max_tokens, "finish_reason": finish_reason, "usage": usage,
               "prompt_chars": len(prompt), "output_chars": len(content), "meta": meta or {}}
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
        with open(stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return content


def ids_from_link(post_link: str) -> tuple[str, str]:
    match = re.search(r"wall-(\d+)_(\d+)", post_link or "")
    if match:
        return match.group(1), match.group(2)
    return "unknown_group", "unknown_post"


def extract_post_text(item: dict) -> str:
    try:
        return (item['main_text'][0]['subtext'][0]['text'] or '').strip()
    except (KeyError, IndexError, TypeError):
        return ''


def extract_comment_text(item: dict) -> str:
    try:
        return (item['subtext'][0]['text'] or '').strip()
    except (KeyError, IndexError, TypeError):
        return ''


def extract_feedback_text(item: dict) -> str:
    try:
        return (item["main_text"][0]["subtext"][0]["text"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


def build_model_text(title: str, review_text: str) -> str:
    safe_title = (title or "").replace('"', '\\"').strip()
    safe_review = " ".join((review_text or "").split()).replace('"', '\\"')
    return f'Наименование: "{safe_title}". Отзыв: "{safe_review}"'


def get_done_ids_numeric(out_dir: str) -> set:
    done = set()
    if not os.path.isdir(out_dir):
        return done
    for name in os.listdir(out_dir):
        m = re.match(r"^result_(\d+)", name)
        if m:
            done.add(m.group(1))
    return done


def get_done_ids_dialog(out_dir: str) -> set:
    done = set()
    if not os.path.isdir(out_dir):
        return done
    for name in os.listdir(out_dir):
        if not name.startswith("result_") or not name.endswith(".json"):
            continue
        core = name[len("result_"):-5]
        if core:
            done.add(core)
    return done


def extract_session_id(link: str) -> str:
    sid = "unknown"
    if not link:
        return sid
    if "id=" in link:
        part = link.split("id=", 1)[1]
        part = part.split("&", 1)[0]
        if part:
            sid = part
    return sid


def parse_dialogues(json_data: dict) -> list:
    res = []
    values = json_data.get("$values", [])
    for item in values:
        link = item.get("Link")
        markup = item.get("Markup") or {}
        mvals = markup.get("$values") or []
        if not mvals:
            continue
        dialog_root = mvals[0].get("Dialog") or {}
        title = dialog_root.get("Title") or ""
        dialogue = dialog_root.get("Dialogue") or {}
        dialog_str = dialogue.get("Dialog") or ""
        if not dialog_str:
            continue
        try:
            messages = json.loads(dialog_str)
        except json.JSONDecodeError:
            print(f"Не удалось распарсить диалог для ссылки {link}", file=sys.stderr)
            continue
        if not messages:
            continue
        nick = messages[0].get("nickname")
        d = []
        for msg in messages:
            nickname = msg.get("nickname")
            role = "Клиент" if nickname == nick else "Психолог"
            p = []
            for s in msg.get("submessages") or []:
                if s.get("type") == "message":
                    text = s.get("text")
                    if text:
                        p.append(text)
            if p:
                full = f"{role}: {' '.join(p)}"
                d.append(full)
        ans_dia = "".join(d) if d else ""
        res.append({
            "link": link,
            "title": title,
            "text": ans_dia
        })
    return res


def split_turns(full_text: str) -> list:
    turns, cur = [], 0
    while cur < len(full_text):
        cpos = full_text.find("Клиент: ", cur)
        ppos = full_text.find("Психолог: ", cur)
        if cpos == -1 and ppos == -1:
            tail = full_text[cur:].strip()
            if tail:
                turns.append(tail)
            break
        nxt = min([x for x in [cpos, ppos] if x != -1])
        if nxt > cur:
            mid = full_text[cur:nxt].strip()
            if mid:
                turns.append(mid)
        if nxt == cpos:
            start = nxt + len("Клиент: ")
            speaker = "Клиент"
        else:
            start = nxt + len("Психолог: ")
            speaker = "Психолог"
        nc = full_text.find("Клиент: ", start)
        np = full_text.find("Психолог: ", start)
        nxt2 = min([x for x in [nc, np] if x != -1], default=-1)
        msg = full_text[start:(nxt2 if nxt2 != -1 else None)].strip()
        turns.append(f"{speaker}: {msg}")
        cur = start + len(msg)
    return turns
