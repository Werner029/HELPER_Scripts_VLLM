import argparse, asyncio, json, os
import aiohttp
from tqdm import tqdm

from vllm_common import gen, extract_feedback_text, build_model_text, get_done_ids_numeric

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

async def process_item(idx, item, session, sem, api_base, model, out_dir, prompt_prefix, max_tokens, done_ids, pbar):
    sid = str(idx)
    if sid in done_ids:
        pbar.update(1)
        return
    title = item["metadata"]["title"]
    raw_text = extract_feedback_text(item)
    if not raw_text:
        pbar.update(1)
        return
    model_text = build_model_text(title, raw_text)
    prompt = prompt_prefix.format(text=model_text)
    ans = await gen(session, sem, api_base, model, prompt, max_tokens=max_tokens,
                    stats_path="run_stats_tags_reviews.jsonl",
                    meta={"task": "feedback_tags", "sid": sid})
    out_path = os.path.join(out_dir, f"result_{sid}_feedback_tags.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump({sid: {"title": title, "text": raw_text, "model_input_text": model_text,
                         "annotated_feedback": ans.strip()}}, fw, ensure_ascii=False, indent=2)
    pbar.update(1)


async def async_main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--input", default="feedbacks_YandexMaps.json")
    p.add_argument("--out", default="out_feedback_tags")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        feedbacks = json.load(f)
    done_ids = get_done_ids_numeric(args.out)
    os.makedirs(args.out, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(feedbacks), desc="tags_reviews")
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[
            process_item(i, item, session, sem, args.api, args.model, args.out,
                         PROMPT_TMPL, args.max_tokens, done_ids, pbar)
            for i, item in enumerate(feedbacks)
        ])
    pbar.close()


def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()