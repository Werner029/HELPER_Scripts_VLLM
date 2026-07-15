import argparse, asyncio, json, os
import aiohttp
from tqdm import tqdm

from vllm_common import gen, ids_from_link, extract_post_text

PROMPT_TMPL = """
Ты - опытный аналитик, специализирующийся на классификации постов в социальных сетях. Тебе подается текст поста, для которого необходимо определить теги (категории). 
Присвой ему любое количество соответствующих смысловых тегов из следующего списка (других
вариантов нет): 
1. Анекдоты и шутки
2. Мемы
3. Жиза/жизненно
4. Новости
5. Политика
6. История
7. Социальное волонтерство
8. Разбор статьи или исследования
9. Интересные факты
10. Цитаты афоризмы
11. Лайфхаки и советы
12. Вопрос аудитории
13. Опрос или голосование
14. Конкурсы
15. Музыка
16. Фото или видео подборка
17. Психология
18. Мотивация
19. Объявление
20. Реклама
21. Другое
Не выбирай слишком много или мало тегов, старайся с их помощью проиллюстрировать категорию поста. Может быть такое, что посту соответствует только один тег. 
Присваивай тег "Другое" только в том случае, если остальные теги не подходят для иллюстрации категории поста. 
Итоговым ответом является строка с названиями тегов через запятую (без пробела).
Пример:
Текст поста: "В подростковом возрасте будущая звезда Голливуда Мэттью Макконахи столкнулся с серьезной проблемой — сильной угревой сыпью. Его мама, которая продавала косметику, предложила ему в качестве лечения масло норки, однако это лишь усугубило ситуацию. Потребовалось два года врачебного вмешательства, чтобы кожа пришла в норму, и результат был настолько впечатляющим, что Мэттью даже получил в школе титул «Самый красивый». \n \nВ это же время его отец планировал подать в суд на производителя мази, требуя компенсацию от 35 до 50 тысяч долларов. Однако окружной прокурор, увидев в школьном альбоме фотографии Макконахи с его новым титулом, закрыл дело, заявив, что юноша уже одержал победу. Так Мэттью не получил денег, но приобрёл нечто более ценное — огромную уверенность в себе. Эта история — яркий пример того, как неудача может обернуться неожиданным триумфом."
Теги: Интересные факты,История,Мотивация
Шаблон для ответа:
Теги: [через запятую без пробелов]
Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот текст поста, проанализируй его и выполни тегирование в соответствии с описанными выше критериями и правилами:  {text}
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
    post_text = extract_post_text(item)
    if not post_text:
        pbar.update(1)
        return
    post_prompt = prompt_prefix.format(text=post_text)
    tags_post = (await gen(session, sem, api_base, model, post_prompt, max_tokens=max_tokens,
                           stats_path="run_stats_tags_posts.jsonl",
                           meta={"task": "tags_post", "group_id": group_id, "post_id": post_id})).strip()
    post_result = {"post_id": post_id, "post_link": post_link, "text": post_text, "tags_post": tags_post}
    groups_result[group_id]["posts"].append(post_result)
    group_out_path = os.path.join(out_dir, f"group_{group_id}_tags_posts.json")
    with open(group_out_path, "w", encoding="utf-8") as fw:
        json.dump(groups_result[group_id], fw, ensure_ascii=False, indent=2)
    pbar.update(1)


async def async_main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--input", default="vk_parsed_groups_1.json")
    p.add_argument("--out", default="out_tags_posts_groups")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    groups_result: dict = {}
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(data), desc="tags_posts")
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[
            process_item(item, session, sem, args.api, args.model, PROMPT_TMPL,
                         args.max_tokens, args.out, groups_result, pbar)
            for item in data
        ])
    pbar.close()
    with open("result_output_tags_posts.json", "w", encoding="utf-8") as fw:
        json.dump(groups_result, fw, ensure_ascii=False, indent=2)


def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()