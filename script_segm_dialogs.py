from __future__ import annotations
import argparse, json, os, sys, requests
from tqdm import tqdm

def foo(json_data):
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

def split_turns(full_text: str):
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

PROMPT_TMPL = """Ты — опытный психолог, специализирующийся на анализе эмоционального состояния клиентов в консультативных диалогах. Тебе подаются сообщения клиента (по одному), который обратился к психологу, а далее у них строится диалог. Твоя задача — разметить реплики клиентов, выделяя фрагменты, которые выражают одну из семи ключевых эмоций (других эмоций нет):
Радость (JOY) — проявление счастья, удовлетворения, надежды, облегчения. Печаль (SADNESS) — грусть, тоска, чувство безнадежности, подавленности. Гнев (ANGER) — раздражение, злость, агрессия, фрустрация. Отвращение (DISGUST) — физическое или моральное отторжение, чувство омерзения Страх (FEAR) — тревога, паника, избегание, страх оценки, будущего. Удивление (SURPRISE) — неожиданные озарения, шок, потрясение. Презрение (CONTEMPT) — чувство превосходства, цинизм, обесценивание .
При этом соблюдай некоторые правила:
Размечай только реплики клиента, пропускай вопросы психолога.
Отмечай не только прямые формулировки ("я боюсь"), но и косвенные признаки ("у меня дрожат руки при мысли о…" => FEAR).
Если эмоция смешанная (например, "злюсь на себя за то, что опять плачу"), используй два тега: «[ANGER]злюсь на себя[/ANGER] [SADNESS]за то, что опять плачу[/SADNESS]».
Учитывай, что в терапии клиенты часто смягчают эмоции (например, "Ну, я немного расстроен" - на деле может быть глубокая печаль).
Размещай только реальные переживания. Например: «Я должен быть счастлив» (это ожидание, а не эмоция) > без разметки. «Я чувствую себя никчемным» => [SADNESS]Я чувствую себя никчемным[/SADNESS].

Для четкой структуры разметки каждую эмоцию заключай в тег:
«[EMOTION]текст[/EMOTION]». Но при этом сохраняй исходный текст без изменений, добавляя только разметку, не используй двойные пробелы. Также не исправляй ошибки в словах, если встречаешь их (если написано «ростройство» не исправляй на «расстройство», оставь так). Одна эмоция может охватывать несколько предложений, если они выражают одну и ту же мысль. Разделяй разметку только при явной смене эмоции. При этом помни, что некоторые отрывки могут не нести под собой никакую эмоцию, следовательно их стоит оставить без разметки. Не размечай фразы типа «Я не понимаю», «Мне странно», «Что со мной?» без дополнительных эмоциональных маркеров.
Исключением является ситуация, в которой контекст или невербальные признаки явно указывают на эмоцию.
Пример:
Клиент: [SADNESS]В последнее время я будто в тумане. Даже простые дела даются с трудом, будто я тону в мелочах.[/SADNESS]
[ANGER]А когда жена спрашивает, почему я такой вялый, меня это бесит![/ANGER] Я ведь и правда не понимаю, что со мной происходит... [FEAR]Страшно даже подумать, что будет дальше…[/FEAR]
Выполни ТОЛЬКО разметку в виде тегов [EMOTION]...[/EMOTION]. Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот сообщение клиента, проанализируй его и выполни разметку в соответствии с описанными выше критериями и правилами:  {text}"""

import json, time, uuid, os, requests

def gen(api_base: str, model: str, prompt: str, max_tokens: int, temperature: float = 0.0, top_p: float = 0.95,
        timeout: float = 1200.0, stats_path: str | None = None, meta: dict | None = None) -> str:
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 20,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    content = j["choices"][0]["message"]["content"]
    usage = j.get("usage")
    finish_reason = j["choices"][0].get("finish_reason")

    if stats_path is not None:
        rec = {
            "ts": time.time(),
            "req_id": str(uuid.uuid4()),
            "model": model,
            "max_tokens": max_tokens,
            "finish_reason": finish_reason,
            "usage": usage,
            "prompt_chars": len(prompt),
            "output_chars": len(content),
            "meta": meta or {},
        }
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
        with open(stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return content


def get_done_ids(out_dir: str):
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

def process_file(path: str, api_base: str, model: str, out_dir: str, prompt_prefix: str, max_tokens: int, done_ids):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dialogues = foo(data)
    result_all = {}
    os.makedirs(out_dir, exist_ok=True)
    for dialogue in tqdm(dialogues, desc=os.path.basename(path)):
        link = dialogue.get("link") or "nolink"
        title = dialogue.get("title") or ""
        full_text = dialogue.get("text") or ""
        sid = extract_session_id(link)
        if sid in done_ids:
            print(f"Пропускаем уже размеченный диалог {link}", file=sys.stderr)
            continue
    #     turns = split_turns(full_text)
    #     annotated = []
    #     for t in turns:
    #         if not t or (not t.startswith("Клиент:")):
    #             continue
    #         prompt = prompt_prefix.format(text=t)
    #         ans = gen(api_base, model, prompt, max_tokens=max_tokens,
    #                   stats_path="run_stats_segm.jsonl",
    #                   meta={"task": "segm", "sid": sid, "file": os.path.basename(path)})
    #         annotated.append(ans.strip())
    #     result_all[link] = {"title": title, "text": full_text, "annotated_dialogue": "\n".join(annotated)}
    #     out_path = os.path.join(out_dir, f"result_{sid}.json")
    #     with open(out_path, "w", encoding="utf-8") as fw:
    #         json.dump({link: result_all[link]}, fw, ensure_ascii=False, indent=2)
    # with open("result_output.json", "w", encoding="utf-8") as fw:
    #     json.dump(result_all, fw, ensure_ascii=False, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api")
    p.add_argument("--model")
    p.add_argument("--out", default="segm_v2_dropout")
    p.add_argument("--files", default='output_markup_test.json')
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()
    done_ids = get_done_ids(args.out)
    for fp in args.files.split():
        if not os.path.exists(fp):
            print(f"Файл {fp} не найден, пропускаем...", file=sys.stderr)
            continue
        process_file(fp, args.api, args.model, args.out, PROMPT_TMPL, args.max_tokens, done_ids)

if __name__ == "__main__":
    main()
