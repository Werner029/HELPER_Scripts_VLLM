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


PROMPT_TMPL = """Ты — опытный психолог, специализирующийся на анализе качества
ответов психологов в консультативных диалогах. Подаваемый тебе
диалог строится следующим образом: первое сообщение
принадлежит клиенту, следующее психологу, каждое сообщение
помечено автором. Твоя задача — необходимо ответить на несколько общих вопросов по
сути диалога: "Насколько клиент в депрессивном состоянии от 1 до
5? (1-отсутствует, 5- сильное)", "Нужна ли психологическая или
психиатрическая помощь от 1 до 5?", "Есть ли растерянность от 1 до
5?", "Есть ли спутанность мышления от 1 до 5?", "Есть ли риск
суицида?" от 1 до 5. Итоговым ответом являются 5 цифр через
запятую (без пробела).
Пример:
«Психолог: Здравствуйте,Александра! Готова с Вам поработать в течение 3х дней в свободном формате.\nЕсли:\n- Вам есть 18 лет\n- Вы осознаете, что это
демоконсультауия, а не полноценная работа с психологом\n- Если от
вас нет ответа в течение суток - имею право закрыть консультацию.
\n- Если консультация будет для вас полезной - прошу оставить
отзыв..Психолог: Я рада Вам. Как лучше обращаться \"ты\" или \"вы\"?\n\nМне интересно с Вами познакомиться.\nРасскажите немного о себе. Сколько Вам лет? Чем
занимаетесь? Есть ли у вас семья? Условия проживания? Друзья?
Хобби? И то, что считаете нужным добавить.\n\nМкня можно
называть Леной и обращаться как вам удобно. (и ты и вы). Психолог:
И ещё, очень важный момент. Чего вы ожидаете от данной
консультации? Какого результата?.Психолог: Здравствуйте, Александра!\nМне приятно с Вами познакомиться! Хочется чтобы Вы были честной и честно
рассказывали о своих чувствах, появляющихся во время
консультации.\n--\nПо запросу - поисследуем ваш страх, возможно,
найдём его причины.\nВ результате,\nстраха может стать меньше и
наступит облегчение.. Психолог: Расскажите, пожалуйста, про
буллинг.\nСколько он длился по времени? В чем заключался?
\nСвязываете ли настоящий страх с буллингом.\nВ университете
такого не было?. Психолог: Какие у Вас отношения с родителями?.
Психолог: Ответа нет от Вас более суток - консультацию закрываю..»
Вопросы: 1,5,3,4,5
Шаблон для ответа:
Вопросы: [5 цифр через запятую]
Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот диалог клиента с психологом, проанализируй его и ответь на вопросы в соответствии с описанными выше критериями и правилами:  {text}"""


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
        if not full_text.strip():
            continue
        prompt = prompt_prefix.format(text=full_text)
        ans = gen(api_base, model, prompt, max_tokens=max_tokens,
                  stats_path="run_stats_quest.jsonl",
                  meta={"task": "quest", "sid": sid, "file": os.path.basename(path)})
        result_all[link] = {
            "title": title,
            "text": full_text,
            "annotated_dialogue": ans.strip(),
        }
        out_path = os.path.join(out_dir, f"result_{sid}_quest.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({link: result_all[link]}, fw, ensure_ascii=False, indent=2)
    with open("result_output.json", "w", encoding="utf-8") as fw:
        json.dump(result_all, fw, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="1")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()
    done_ids = get_done_ids(args.out)
    for fp in args.files:
        if not os.path.exists(fp):
            print(f"Файл {fp} не найден, пропускаем...", file=sys.stderr)
            continue
        process_file(fp, args.api, args.model, args.out, PROMPT_TMPL, args.max_tokens, done_ids)


if __name__ == "__main__":
    main()
