import argparse, asyncio, json, os
import aiohttp
from tqdm import tqdm

from vllm_common import gen, extract_feedback_text, build_model_text, get_done_ids_numeric

PROMPT_TMPL = """Ты — опытный психолог, специализирующийся на анализе тональности в отзывах о местах и организациях.
В подаваемом тебе тексте содержится наименование места, на которое оставлен отзыв, а далее сам текст отзыва.
Твоя задача — необходимо определить тональность текста отзыва: "нейтральная", "позитивная", "негативная".
Отзыву можно присвоить только одну из перечисленных тональностей. 
В формате вывода "Позитивная" тональность задается цифрами [1,1,1,1,1], "Нейтральная" - [3,3,3,3,3], "Негативная" - [5,5,5,5,5]. 
Итоговым ответом являются 5 одинаковых цифр через запятую (без пробела).
Пример:
«Наименование: "Продукты Ермолино". Отзыв: "Замечательная сеть магазинов в общем, хороший ассортимент, цены приемлемые, а главное качество на высоте!!! Спасибо тем, кто открыл сеть этих магазинчиков!!!!"»
Тональность: 1,1,1,1,1
Шаблон для ответа:
Тональность: [5 одинаковых цифр, соответствующих тональности, через запятую без пробела]
Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот отзыв, проанализируй его и ответь на вопросы в соответствии с описанными выше критериями и правилами:  {text}"""


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
                    stats_path="run_stats_tone_reviews.jsonl",
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
    pbar = tqdm(total=len(feedbacks), desc="tone_reviews")
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