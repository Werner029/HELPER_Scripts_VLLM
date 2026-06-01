from __future__ import annotations
import argparse, json, os, sys, requests, re
from tqdm import tqdm
import time, uuid

PROMPT_TMPL = """
Ты — опытный аналитик данных, специализирующийся на классификации отзывов о местах и организациях. В подаваемом тебе тексте содержится наименование места, на которое оставлен отзыв, а далее сам текст отзыва.
Твоя задача — присвоить отзыву определенные теги.
Присвой ему любое количество соответствующих смысловых тегов из следующего списка (других
вариантов нет): 1. Кафе и ресторан, 2. Продуктовый магазин, 3. Магазин товаров, 4. Аптека,
5. Медицина, 6. Салон красоты, 7. Спа и баня, 8. Спорт и фитнес, 9. Кино и театр, 10. Музей и выставки, 11. Парк развлечений,
12. Ночной клуб, 13. Образование, 14. Детские учреждения, 15. Автосервис, 16. АЗС, 17. Парковка, 18. Автосалон,
19. Отели и гостиницы, 20. Банки и финансы, 21. Бытовые услуги, 22. Оптика, 23. Торговый центр,
24. Парки и скверы, 25. Религиозные места, 26. Другое \n
Не выбирай слишком много или мало тегов, старайся с их
помощью проиллюстрировать категорию места, на который оставлен отзыв. Может быть такое, что отзыву соответствует только один тег. Присваивай тег "Другое" только в том случае, если остальные теги не подходят для иллюстрации категории места.
Итоговым ответом является строка с названиями тегов через запятую (без пробела).
Пример:
«Наименование: "Продукты Ермолино". Отзыв: "Замечательная сеть магазинов в общем, хороший ассортимент, цены приемлемые, а главное качество на высоте!!! Спасибо тем, кто открыл сеть этих магазинчиков!!!!"»
Теги: Продуктовый магазин
Шаблон для ответа:
Теги: [через запятую без пробелов]
Теперь проанализируй отзыв ниже и выполни тегирование этого отзыва
Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот отзыв, проанализируй его и выполни тегирование в соответствии с описанными выше критериями и правилами:  {text}"""

def extract_feedback_text (item: dict) -> str : # извлекает чистый текст отзыва из одного объекта отзыва
    try:
        return (item["main_text"][0]["subtext"][0]["text"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""

def build_model_text (title: str, review_text: str) -> str: # формирует строку в формате "Наименование:... Отзыв:..."
    safe_title = (title or "").replace('"', '\\"').strip()
    safe_review = " ".join((review_text or "").split()).replace('"', '\\"')
    return f'Наименование: "{safe_title}". Отзыв: "{safe_review}"'

def gen(api_base: str, model: str, prompt: str, max_tokens: int, temperature: float = 0.0, top_p: float = 0.95,
        timeout: float = 1200.0, stats_path: str | None = None, meta: dict | None = None) -> str: #
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
        m = re.match(r"^result_(\d+)", name)
        if m:
            done.add(m.group(1))
    return done

def process_file(path: str, api_base: str, model: str, out_dir: str, prompt_prefix: str, max_tokens: int, done_ids):
    with open(path, "r", encoding="utf-8") as f:
        feedbacks = json.load(f)
    result_all = {}
    os.makedirs(out_dir, exist_ok=True)
    for idx, item in enumerate(tqdm(feedbacks, desc=os.path.basename(path))):
        sid = str(idx) # нумерация по индексам отзывов
        title = item["metadata"]["title"]
        raw_text = extract_feedback_text(item)
        if not raw_text:
            continue
        model_text = build_model_text(title, raw_text)
        prompt = prompt_prefix.format(text=model_text)
        ans = gen(api_base, model, prompt, max_tokens=max_tokens,
                  stats_path="run_stats_feedback_tags.jsonl",
                  meta={"task": "feedback_tags", "sid": sid, "file": os.path.basename(path)})
        result_all[sid] = {
            "title": title,
            "text": raw_text,
            "model_input_text": model_text,
            "annotated_feedback": ans.strip(),
        }
        out_path = os.path.join(out_dir, f"result_{sid}_feedback_tags.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({sid: result_all[sid]}, fw, ensure_ascii=False, indent=2)
    with open("result_output_feedback_tags.json", "w", encoding="utf-8") as fw:
        json.dump(result_all, fw, ensure_ascii=False, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--input", default="feedbacks_YandexMaps.json")
    p.add_argument("--out", default="out_feedback_tags")
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()

    done_ids = get_done_ids(args.out)
    process_file(args.input, args.api, args.model, args.out, PROMPT_TMPL, args.max_tokens, done_ids)

if __name__ == "__main__":
    main()