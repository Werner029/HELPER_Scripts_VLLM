import argparse, asyncio, json, os
import aiohttp
from tqdm import tqdm

from vllm_common import gen, ids_from_link, extract_post_text, extract_comment_text

PROMPT_TMPL = """
Ты — опытный психолог, специализирующийся на эмоциональной сегментации текстов в социальных сетях. Тебе подается один текст (это либо пост, либо комментарий). Твоя задача — разметить фрагменты текста, выделяя части, которые выражают одну из семи ключевых эмоций (других эмоций нет):
Радость (JOY) — положительное эмоциональное состояние, возникающее при восприятии благоприятного события, соответствующего желаниям и ожиданиям(проявление счастья, яркого удовольствия, облегчения, воодушевления). 
Печаль (SADNESS) — отрицательное эмоциональное состояние, связанное с утратой или неблагоприятным событием, которое противоречит желаниям, вызывает чувство потери или невозможности (грусть, тоска, чувство безнадежности, подавленности, неудовольствия, спад мотивации, фрустрация). 
Удовлетворение (SATISFACTION)- удовольствие от приятного подтвердившегося события, положительное состояние от подтверждения ожиданий, возникающее при реализации ожидаемого благоприятного события (спокойное удовольствие, ощущение завершенности, соответствие ожиданиям).
Разочарование (DISAPPOINTMENT)- неудовольствие от не подтвердившегося приятного события, отрицательное состояние от несоответствия ожиданий, когда ожидаемое благоприятное событие не реализовалось (несоответствие ожиданиям, чувство неудовлетворенности).
Стыд (SHAME) - отрицательная оценка своих действий или своей личности, несоответствие собственных действий нормам или ожиданиям (робость, застенчивость, стыдливость, самокритика, смущение, желание скрыться).
Гордость (PRIDE) - положительная оценка своих действий или собственной личности, собственные действия соответствуют или превосходят нормы и ожидания (уверенность, удовлетворение собой, чувство достижения).
Отвращение (DISGUST) — отрицательная эмоция отторжения объекта или ситуации, восприятие объекта как неприятного или недопустимого (физическое или моральное отторжение, чувство омерзения, избегание объекта).
Симпатия (SYMPATHY) - приятное влечение к объекту, положительное эмоциональное отношение к объекту или человеку, объект воспринимается как приятный, безопасный, привлекательный (интерес, расположение, стремление к сближению).
Страх (FEAR) — неудовольствие от неприятного предполагаемого события, ожидание угрозы, восприятие возможного события как опасного или нежелательного (тревога, паника, избегание события, страх будущего, напряжение).
Надежда (HOPE) - удовольствие от предполагаемого приятного события, возможное событие воспринимается как желаемое и достижимое (ожидание события, воодушевление, оптимизм по отношению к будущему).
Презрение (CONTEMPT) — отрицательная оценка чужих действий или качеств, действия другого воспринимаются как недостойные или ниже нормы (чувство превосходства, цинизм, обесценивание, дистанцирование от другого).
Восхищение (ADMIRATION) - положительная оценка чужих действий, действия другого воспринимаются как выдающиеся или превышающие норму (уважение, признание, стремление поддержать)
Удивление (SURPRISE) — нейтрально-валентная эмоция, возникающая при неожиданном событии (неожиданные озарения, шок, потрясение, резкое переключение внимания). 
Гнев (ANGER) — сложная отрицательная эмоция, комбинация презрения и печали (раздражение, злость, агрессия, стремление к восстановлению справедливости). 
При этом соблюдай некоторые правила:
Отмечай не только прямые формулировки ("я боюсь"), но и косвенные признаки ("у меня дрожат руки при мысли о…" => FEAR).
Если эмоция смешанная (например, "злюсь на себя за то, что опять плачу"), используй два тега: «[ANGER]злюсь на себя[/ANGER] [SADNESS]за то, что опять плачу[/SADNESS]».
Учитывай, что люди часто смягчают эмоции (например, "Ну, я немного расстроен" - на деле может быть глубокая печаль).
Размечай только реальные переживания. Например: «Он должен быть счастлив» (это ожидание, а не эмоция) => без разметки. «Я чувствую себя никчемным на фоне него» => [SADNESS]Я чувствую себя никчемным на фоне него[/SADNESS].

Для четкой структуры разметки каждую эмоцию заключай в тег:
«[EMOTION]текст[/EMOTION]». Но при этом сохраняй исходный текст без изменений, добавляя только разметку, не используй двойные пробелы. Также не исправляй ошибки в словах, если встречаешь их (если написано «ростройство» не исправляй на «расстройство», оставь так). 
Одна эмоция может охватывать несколько предложений, если они выражают одну и ту же мысль. Разделяй разметку только при явной смене эмоции. При этом помни, что некоторые отрывки могут не нести под собой никакую эмоцию, следовательно их стоит оставить без разметки. 
Не размечай фразы типа «Я не понимаю», «Мне странно», «Что это?» без дополнительных эмоциональных маркеров.
Исключением является ситуация, в которой контекст или невербальные признаки явно указывают на эмоцию.
Пример:
Текст: "Страшно даже подумать, что будет дальше..."
Разметка: [FEAR]Страшно даже подумать, что будет дальше...[/FEAR]
Выполни ТОЛЬКО разметку текста в виде тегов [EMOTION]...[/EMOTION]. Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот текст, проанализируй его и выполни разметку в соответствии с описанными выше критериями и правилами:  {text}"""

CONCURRENCY = 16


async def process_item(
    item: dict,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    api_base: str, model: str,
    prompt_prefix: str, max_tokens: int,
    out_dir: str,
    groups_result: dict,
    pbar: tqdm,
) -> None:
    if item.get('type') != 'post':
        pbar.update(1)
        return

    source = item['source']
    metadata = item['metadata']
    post_link = source['link']
    group_id, post_id = ids_from_link(post_link)
    group_title = metadata.get('title')

    if group_id not in groups_result:
        groups_result[group_id] = {
            'group_id': group_id,
            'group_title': group_title,
            'posts': []
        }

    post_text = extract_post_text(item)
    if not post_text:
        pbar.update(1)
        return

    tasks = []
    post_prompt = prompt_prefix.format(text=post_text)
    tasks.append(gen(session, sem, api_base, model, post_prompt, max_tokens=max_tokens,
                     stats_path="run_stats_segm_posts.jsonl",
                     meta={"task": "segm_post", "group_id": group_id, "post_id": post_id}))
    comment_items = []
    for comment in item['comments']:
        comment_text = extract_comment_text(comment)
        if not comment_text:
            continue
        comment_id = str(comment["comment_id"])
        comment_prompt = prompt_prefix.format(text=comment_text)
        tasks.append(gen(session, sem, api_base, model, comment_prompt, max_tokens=max_tokens,
                         stats_path="run_stats_segm_posts.jsonl",
                         meta={"task": "segm_comment", "group_id": group_id,
                               "post_id": post_id, "comment_id": comment_id}))
        comment_items.append((comment_id, comment["link"], comment_text))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    annotated_post = results[0].strip() if isinstance(results[0], str) else ""
    comments_result = []
    for i, (cid, clink, ctxt) in enumerate(comment_items):
        ann = results[i + 1]
        comments_result.append({
            "comment_id": cid,
            "comment_link": clink,
            "text": ctxt,
            "annotated_comment": ann.strip() if isinstance(ann, str) else "",
        })

    post_result = {
        "post_id": post_id,
        "post_link": post_link,
        "text": post_text,
        "annotated_post": annotated_post,
        "comments": comments_result
    }

    groups_result[group_id]['posts'].append(post_result)
    group_out_path = os.path.join(out_dir, f"group_{group_id}_segm.json")
    with open(group_out_path, "w", encoding="utf-8") as fw:
        json.dump(groups_result[group_id], fw, ensure_ascii=False, indent=2)

    pbar.update(1)


async def async_main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--input", default="vk_parsed_groups_1.json")
    p.add_argument("--out", default="out_segm_groups")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    groups_result: dict = {}

    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(data), desc="segm_posts")

    async with aiohttp.ClientSession() as session:
        coros = [
            process_item(item, session, sem, args.api, args.model,
                         PROMPT_TMPL, args.max_tokens, args.out,
                         groups_result, pbar)
            for item in data
        ]
        await asyncio.gather(*coros)

    pbar.close()

    with open('result_output_segm_posts.json', 'w', encoding="utf-8") as fw:
        json.dump(groups_result, fw, ensure_ascii=False, indent=2)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()