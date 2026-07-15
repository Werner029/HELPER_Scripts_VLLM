import argparse, asyncio, json, os
import aiohttp
from tqdm import tqdm

from vllm_common import gen, ids_from_link, extract_comment_text

PROMPT_TMPL = """
Ты - опытный психолог, специализирующийся на анализе тональности комментариев в социальных сетях. Тебе подается текст комментария, для которого необходимо определить тональность. 
Твоя задача определить тональность текста комментария: "нейтральная", "позитивная", "негативная", "не ясно". Комментарию можно присвоить только одну из перечисленных тональностей.
В формате вывода "Позитивная" тональность задается цифрами [1,1,1,1,1], "Нейтральная" - [3,3,3,3,3], "Негативная" - [5,5,5,5,5], "Не ясно" - [0,0,0,0,0].
Итоговым ответом являются 5 одинаковых цифр через запятую (без пробела). Тональность "не ясно" присваивается в исключительных случаях, когда когда определить тональность невозможно - это касается ситуаций, когда комментарий полностью лишен смыслового контекста.
Пример: 
Текст: "Очень интересный пост!"
Тональность: 1,1,1,1,1
Шаблон для ответа:
Тональность: [5 одинаковых цифр, соответствующих тональности, через запятую без пробела]
Не добавляй никаких пояснений, примечаний или блоков <think>. 
Вот комментарий, проанализируй его и присвой тональность в соответствии с описанными выше критериями и правилами: {text}
"""

async def process_item(item, session, sem, api_base, model, prompt_prefix, max_tokens, out_dir, groups_result, pbar):
    if item.get('type') != 'post':
        pbar.update(1)
        return
    source = item['source']
    metadata = item['metadata']
    post_link = source['link']
    group_id, post_id = ids_from_link(post_link)
    group_title = metadata.get('title')
    if group_id not in groups_result:
        groups_result[group_id] = {'group_id': group_id, 'group_title': group_title, 'posts': []}

    comment_items = []
    for comment in item['comments']:
        comment_text = extract_comment_text(comment)
        if not comment_text:
            continue
        comment_id = str(comment["comment_id"])
        comment_link = comment["link"]
        comment_prompt = prompt_prefix.format(text=comment_text)
        comment_items.append((comment_id, comment_link, comment_text, comment_prompt))

    results = await asyncio.gather(*[
        gen(session, sem, api_base, model, cp, max_tokens=max_tokens,
            stats_path="run_stats_tone_comments.jsonl",
            meta={"task": "tone_comment", "group_id": group_id, "post_id": post_id,
                  "comment_id": cid})
        for cid, _, _, cp in comment_items
    ], return_exceptions=True)

    comments_result = []
    for i, (cid, clink, ctxt, _) in enumerate(comment_items):
        ann = results[i]
        comments_result.append({
            "comment_id": cid, "comment_link": clink,
            "text": ctxt, "comment_tone": ann.strip() if isinstance(ann, str) else ""
        })

    post_result = {"post_id": post_id, "post_link": post_link, "comments": comments_result}
    groups_result[group_id]["posts"].append(post_result)
    group_out_path = os.path.join(out_dir, f"group_{group_id}_tone_comments.json")
    with open(group_out_path, "w", encoding="utf-8") as fw:
        json.dump(groups_result[group_id], fw, ensure_ascii=False, indent=2)
    pbar.update(1)


async def async_main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--input", default="vk_parsed_groups_1.json")
    p.add_argument("--out", default="out_tone_comments_groups")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    groups_result: dict = {}
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(data), desc="tone_comments")
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[
            process_item(item, session, sem, args.api, args.model, PROMPT_TMPL,
                         args.max_tokens, args.out, groups_result, pbar)
            for item in data
        ])
    pbar.close()
    with open("result_output_tone_comments.json", "w", encoding="utf-8") as fw:
        json.dump(groups_result, fw, ensure_ascii=False, indent=2)


def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()