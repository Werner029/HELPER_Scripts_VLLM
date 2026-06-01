from __future__ import annotations
import argparse, json, os, sys, requests, re
from tqdm import tqdm
import time, uuid

PROMPT_TMPL = """Ты — опытный психолог, специализирующийся на анализе эмоционального состояния людей в отзывах на места и организации. В подаваемом тебе тексте содержится наименование места, на которое оставлен отзыв, а далее сам текст отзыва. Твоя задача — разметить текст отзыва, выделяя фрагменты, которые выражают одну из семи ключевых эмоций (других эмоций нет):
Радость (JOY) — проявление счастья, удовлетворения, надежды, облегчения. Печаль (SADNESS) — грусть, тоска, чувство безнадежности, подавленности. Гнев (ANGER) — раздражение, злость, агрессия, фрустрация. Отвращение (DISGUST) — физическое или моральное отторжение, чувство омерзения. Страх (FEAR) — тревога, паника, избегание, страх оценки, будущего. Удивление (SURPRISE) — неожиданные озарения, шок, потрясение. Презрение (CONTEMPT) — чувство превосходства, цинизм, обесценивание .
При этом соблюдай некоторые правила:
Размечай только текст отзыва, размечать наименование места не требуется.
Отмечай не только прямые формулировки ("я ненавижу"), но и косвенные признаки ("ужасное качество!" => ANGER).
Если эмоция смешанная (например, "боюсь давать эту еду детям, потому что готовят непонятно как!"), используй два тега: «[FEAR]боюсь давать эту еду детям[/FEAR], потому что [ANGER]готовят непонятно как![/ANGER]».
Учитывай, что люди часто смягчают эмоции (например, "Ну, я немного расстроен" - на деле может быть глубокая печаль).
Размечай только реальные переживания. Например: «Надеюсь, владелец примет это во внимание» (это ожидание, а не эмоция) > без разметки. «Я ожидал лучшего» => [SADNESS]Я ожидал лучшего[/SADNESS].

Для четкой структуры разметки каждую эмоцию заключай в тег:
«[EMOTION]текст[/EMOTION]». Но при этом сохраняй исходный текст без изменений, добавляя только разметку, не используй двойные пробелы. Также не исправляй ошибки в словах, если встречаешь их (если написано «ростройство» не исправляй на «расстройство», оставь без изменений). 
Одна эмоция может охватывать несколько предложений, если они выражают одну и ту же мысль. Разделяй разметку только при явной смене эмоции. При этом помни, что некоторые отрывки могут не нести под собой эмоцию, следовательно их стоит оставить без разметки. Не размечай фразы типа «Я не понимаю», «Мне странно», «Что это?» без дополнительных эмоциональных маркеров.
Исключением является ситуация, в которой контекст или невербальные признаки явно указывают на эмоцию.
Пример:
"Наименование: "LimeFit". Отзыв: "Не знаю смутят ли кого-то данные правила, но я была удивлена: \\n1. Хочешь что бы твой шкаф замыкался - купи замочек\\n2. Ты должен предоставить свой отпечаток пальца (полнейшая дичь) \\n3. Ставят подпись на договоре с клиентом по доверенности , графу с номером доверенности оставляют пустой , а на вопрос о номере доверенности говорят номер «2»\\nВы серьезно? Номер 2? \\nПредоставить доверенность не могут, но говорят что у них в клубе «свои» доверенности, типа особенные какие-то \\nЦирк.\n"
Разметка: Не знаю смутят ли кого-то данные правила, но [SURPRISE]я была удивлена[/SURPRISE]: \\n
1. [ANGER]Хочешь что бы твой шкаф замыкался - купи замочек[/ANGER]\\n
2. [ANGER]Ты должен предоставить свой отпечаток пальца (полнейшая дичь)[/ANGER] \\n
3. Ставят подпись на договоре с клиентом по доверенности , графу с номером доверенности оставляют пустой , а [ANGER]на вопрос о номере доверенности говорят номер «2»\\nВы серьезно? Номер 2?[/ANGER] \\n
[ANGER]Предоставить доверенность не могут, но говорят что у них в клубе «свои» доверенности, типа особенные какие-то[/ANGER] \\n[ANGER]Цирк[/ANGER].\n

Выполни ТОЛЬКО разметку в виде тегов [EMOTION]...[/EMOTION]. Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот отзыв, проанализируй его и выполни разметку эмоциональной сегментации в соответствии с описанными выше критериями и правилами:  {text}"""


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