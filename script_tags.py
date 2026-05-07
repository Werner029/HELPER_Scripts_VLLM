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


PROMPT_TMPL = """
Ты — опытный психолог, специализирующийся на анализе качества
ответов психологов в консультативных диалогах. Подаваемый тебе
диалог строится следующим образом: первое сообщение
принадлежит клиенту, следующее психологу, каждое сообщение
помечено автором. Твоя задача — присвоить диалогу определенные теги.
Присвой ему любое количество соответствующих смысловых тегов из следующего списка (других
вариантов нет): 1. Депрессия и тревожные расстройства, 2.
Личностные кризисы и самооценка, 3. Семейные и партнёрские
отношения, 4. Психосоматика и здоровье, 5. Зависимости и
компульсивное поведение, 6. Травмы и посттравматические
состояния, 7. Социальная адаптация и конфликты, 8. Карьера,
финансы и профессиональные трудности, 9. Дети и подростки:
воспитание и проблем, 10. Нарушения пищевого поведения, 11.
Сексуальность и интимные проблемы, 12. Неврозы и эмоциональная
нестабильность, 13. Принятие решений и самореализация, 14.
Необходимость психиатрической помощи, 15. Экзистенциальные
кризисы. Не выбирай слишком много или мало тегов, старайся с их
помощью охватить специфику диалога. Итоговым ответом является
строка с названиями тегов через запятую (без пробела).
Пример:
«Клиент: Здравствуйте!\nДело в том, что у меня в жизни было много
длительных стрессов.\nНа данный момент я чувствую себя намного
счастливее.\nИ я боюсь быть счастливой,потому что мне страшно ,что
мое счастье закончится, и начинается дикая тревога от страха потери
или от страха быть свободной и счастливой ,якобы это все не
надолго,по поводу того,что может произойти что-то плохое.\nНу и это
счастье во многом связано с тем,что в моей жизни многое изменилось
в лучшую сторону и я в это не верю.. Психолог: Здравствуйте,
Александра! Готова с Вам поработать в течение 3х дней в свободном
формате.\nЕсли:\n- Вам есть 18 лет\n- Вы осознаете, что это
демоконсультауия, а не полноценная работа с психологом\n- Если от
вас нет ответа в течение суток - имею право закрыть консультацию.
\n- Если консультация будет для вас полезной - прошу оставить
отзыв.. Клиент: Хорошо. Психолог: Я рада Вам. Как лучше
обращаться \"ты\" или \"вы\"?\n\nМне интересно с Вами
познакомиться.\nРасскажите немного о себе. Сколько Вам лет? Чем
занимаетесь? Есть ли у вас семья? Условия проживания? Друзья?
Хобби? И то, что считаете нужным добавить.\n\nМкня можно
называть Леной и обращаться как вам удобно. (и ты и вы). Психолог:
И ещё, очень важный момент. Чего вы ожидаете от данной
консультации? Какого результата?. Клиент: Здравствуйте, у меня
время +6 к Мск, поэтому позже отвечаю\nОбращаться можете на
вы\nЛена,спасибо вам,что отвечаете мне))\nМеня зовут Александра,
мне 25 лет,я учусь на переводчика английского и немного
преподаю\nУ меня есть хобби-рисование,читаю книги,смотрю
фильмы\nДело в том,что я раньше очень много страдала от буллинга
в школе, очень много было переживаний\nСейчас я на 4-ом курсе
института и ,понимаете, я не верю,что скоро я закончу вуз, еще есть
5-ый курс\nИ у меня появился парень из другой страны\nИ я не верю
в то,что ,возможно скоро мы встретимся и т.д.\nОт консультации
желаю получить облегчение от тревоги,страха быть
счастливой,понять вообще почему я боюсь этого\nПотому что я
боюсь,что я заболею или не встречусь с любимым и т.д.
\nПредставляю негативные картинки. Клиент: Я живу на данный
момент с родителями,пока еще нет полной финансовой
независимости. Психолог: Здравствуйте, Александра!\nМне приятно с
Вами познакомиться! Хочется чтобы Вы были честной и честно
рассказывали о своих чувствах, появляющихся во время
консультации.\n--\nПо запросу - поисследуем ваш страх, возможно,
найдём его причины.\nВ результате,\nстраха может стать меньше и
наступит облегчение.. Психолог: Расскажите, пожалуйста, про
буллинг.\nСколько он длился по времени? В чем заключался?
\nСвязываете ли настоящий страх с буллингом.\nВ университете
такого не было?. Психолог: Какие у Вас отношения с родителями?.
Психолог: Ответа нет от Вас более суток - консультацию закрываю..»
Теги: Семейные и партнёрские отношения, Социальная адаптация и конфликты, Травмы и посттравматические
состояния
Шаблон для ответа:
Теги: [через запятую без пробелов]
Теперь проанализируй диалог ниже и выполни тегирование этого диалога
Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот диалог, проанализируй его и выполни тегирование в соответствии с описанными выше критериями и правилами:  {text}"""
import json, time, uuid, os, requests

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
                  stats_path="run_stats_tags.jsonl",
                  meta={"task": "tags", "sid": sid, "file": os.path.basename(path)})
        result_all[link] = {
            "title": title,
            "text": full_text,
            "annotated_dialogue": ans.strip(),
        }
        out_path = os.path.join(out_dir, f"result_{sid}_tags.json")
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
