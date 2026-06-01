from __future__ import annotations
import argparse, json, os, sys, requests, re
from tqdm import tqdm
import time, uuid

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

def ids_from_link (post_link: str) -> tuple[str, str]: # извлекает id группы и id поста из ссылки
    match = re.search(r"wall-(\d+)_(\d+)", post_link or "")
    if match:
        return match.group(1), match.group(2)
    return "unknown_group", "unknown_post"

def extract_post_text (item: dict) -> str:
    try:
        return (item['main_text'][0]['subtext'][0]['text'] or '').strip()
    except(KeyError, IndexError, TypeError):
        return ''

def extract_comment_text (item: dict) -> str:
    try:
        return (item['subtext'][0]['text'] or '').strip()
    except(KeyError, IndexError, TypeError):
        return ''

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

def process_file(path: str, api_base: str, model: str, out_dir: str, prompt_prefix: str, max_tokens: int):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    groups_result = {}
    for item in tqdm(data, desc = os.path.basename(path)):
        if item.get('type') != 'post':
            continue

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
            continue
        post_prompt = prompt_prefix.format(text=post_text)
        annotated_post = gen(
            api_base=api_base,
            model=model,
            prompt=post_prompt,
            max_tokens=max_tokens,
            stats_path="run_stats_segm_posts.jsonl",
            meta={"task": "segm_post", "group_id": group_id, "post_id": post_id}
        ).strip()

        comments_result = []
        for comment in item['comments']:
            comment_text = extract_comment_text(comment)
            if not comment_text:
                continue
            comment_id = str(comment["comment_id"])
            comment_link = comment["link"]
            comment_prompt = prompt_prefix.format(text=comment_text)

            annotated_comment = gen(
                api_base=api_base,
                model=model,
                prompt=comment_prompt,
                max_tokens=max_tokens,
                stats_path="run_stats_segm_posts.jsonl",
                meta={
                    "task": "segm_comment",
                    "group_id": group_id,
                    "post_id": post_id,
                    "comment_id": comment_id
                }
            ).strip()

            comments_result.append({
                "comment_id": comment_id,
                "comment_link": comment_link,
                "text": comment_text,
                "annotated_comment": annotated_comment
            })

        post_result ={
            "post_id": post_id,
            "post_link": post_link,
            "text": post_text,
            "annotated_post": annotated_post,
            "comments": comments_result
        }

        groups_result[group_id]['posts'].append(post_result)
        group_out_path = os.path.join(out_dir, f"group_{group_id}_segm.json")
        with open(group_out_path, "w", encoding="utf-8") as fw:
            json.dump(groups_result[group_id], fw, ensure_ascii=False, indent = 2)
    with open('result_output_segm_posts.json', 'w', encoding="utf-8") as fw:
        json.dump(groups_result, fw, ensure_ascii=False, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--input", default="vk_parsed_groups_1.json")
    p.add_argument("--out", default="out_segm_groups")
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()

    process_file(args.input, args.api, args.model, args.out, PROMPT_TMPL, args.max_tokens)

if __name__ == "__main__":
    main()

