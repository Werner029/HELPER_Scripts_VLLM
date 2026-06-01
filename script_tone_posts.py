from __future__ import annotations
import argparse, json, os, sys, requests, re
from tqdm import tqdm
import time, uuid

PROMPT_TMPL = """
Ты - опытный психолог, специализирующийся на анализе тональности постов в социальных сетях. Тебе подается текст поста, для которого необходимо определить тональность. 
Твоя задача определить тональность текста поста: "нейтральная", "позитивная", "негативная". Посту можно присвоить только одну из перечисленных тональностей.
В формате вывода "Позитивная" тональность задается цифрами [1,1,1,1,1], "Нейтральная" - [3,3,3,3,3], "Негативная" - [5,5,5,5,5].
Итоговым ответом являются 5 одинаковых цифр через запятую (без пробела).
Пример: 
Текст: "В подростковом возрасте будущая звезда Голливуда Мэттью Макконахи столкнулся с серьезной проблемой — сильной угревой сыпью. Его мама, которая продавала косметику, предложила ему в качестве лечения масло норки, однако это лишь усугубило ситуацию. Потребовалось два года врачебного вмешательства, чтобы кожа пришла в норму, и результат был настолько впечатляющим, что Мэттью даже получил в школе титул «Самый красивый». \n \nВ это же время его отец планировал подать в суд на производителя мази, требуя компенсацию от 35 до 50 тысяч долларов. Однако окружной прокурор, увидев в школьном альбоме фотографии Макконахи с его новым титулом, закрыл дело, заявив, что юноша уже одержал победу. Так Мэттью не получил денег, но приобрёл нечто более ценное — огромную уверенность в себе. Эта история — яркий пример того, как неудача может обернуться неожиданным триумфом."
Тональность: 1,1,1,1,1
Шаблон для ответа:
Тональность: [5 одинаковых цифр, соответствующих тональности, через запятую без пробела]
Не добавляй никаких пояснений, примечаний или блоков <think>. 
Вот текст поста, проанализируй его и присвой тональность в соответствии с описанными выше критериями и правилами: {text}
"""

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
        tone_post = gen(
            api_base=api_base,
            model=model,
            prompt=post_prompt,
            max_tokens=max_tokens,
            stats_path="run_stats_tone_posts.jsonl",
            meta={"task": "tone_post", "group_id": group_id, "post_id": post_id}
        ).strip()

        post_result = {
            "post_id": post_id,
            "post_link": post_link,
            "text": post_text,
            "tone_post": tone_post
        }
        groups_result[group_id]["posts"].append(post_result)
        group_out_path = os.path.join(out_dir, f"group_{group_id}_tone_posts.json")
        with open(group_out_path, "w", encoding="utf-8") as fw:
            json.dump(groups_result[group_id], fw, ensure_ascii=False, indent=2)
    with open("result_output_tone_posts.json", "w", encoding="utf-8") as fw:
        json.dump(groups_result, fw, ensure_ascii=False, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--input", default="vk_parsed_groups_1.json")
    p.add_argument("--out", default="out_tone_posts_groups")
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()

    process_file(args.input, args.api, args.model, args.out, PROMPT_TMPL, args.max_tokens)

if __name__ == "__main__":
    main()