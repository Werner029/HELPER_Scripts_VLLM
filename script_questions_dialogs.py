import argparse, asyncio, json, os, sys
import aiohttp
from tqdm import tqdm

from vllm_common import gen, get_done_ids_dialog, extract_session_id, parse_dialogues


PROMPT_TMPL = """Ты — опытный психолог, специализирующийся на анализе качества
ответов психологов в консультативных диалогах. Подаваемый тебе
диалог строится следующим образом: первое сообщение
принадлежит клиенту, следующее психологу, каждое сообщение
помечено автором. Твоя задача — необходимо ответить на несколько общих вопросов по
сути диалога: "Насколько клиент в депрессивном состоянии от 1 до
5? (1-отсутствует, 5- сильное)", "Нужна ли психологическая или
психиатрическая помощь от 1 до 5?", "Есть ли растерянность от 1 до
5?", "Есть ли спутанность мышления от 1 до 5?", "Есть ли риск
суицида?" от 1 до 5. Итоговым ответом являются 5 цифр через
запятую (без пробела).
Пример:
«Психолог: Здравствуйте,Александра! Готова с Вам поработать в течение 3х дней в свободном формате.\nЕсли:\n- Вам есть 18 лет\n- Вы осознаете, что это
демоконсультауия, а не полноценная работа с психологом\n- Если от
вас нет ответа в течение суток - имею право закрыть консультацию.
\n- Если консультация будет для вас полезной - прошу оставить
отзыв..Психолог: Я рада Вам. Как лучше обращаться \"ты\" или \"вы\"?\n\nМне интересно с Вами познакомиться.\nРасскажите немного о себе. Сколько Вам лет? Чем
занимаетесь? Есть ли у вас семья? Условия проживания? Друзья?
Хобби? И то, что считаете нужным добавить.\n\nМкня можно
называть Леной и обращаться как вам удобно. (и ты и вы). Психолог:
И ещё, очень важный момент. Чего вы ожидаете от данной
консультации? Какого результата?.Психолог: Здравствуйте, Александра!\nМне приятно с Вами познакомиться! Хочется чтобы Вы были честной и честно
рассказывали о своих чувствах, появляющихся во время
консультации.\n--\nПо запросу - поисследуем ваш страх, возможно,
найдём его причины.\nВ результате,\nстраха может стать меньше и
наступит облегчение.. Психолог: Расскажите, пожалуйста, про
буллинг.\nСколько он длился по времени? В чем заключался?
\nСвязываете ли настоящий страх с буллингом.\nВ университете
такого не было?. Психолог: Какие у Вас отношения с родителями?.
Психолог: Ответа нет от Вас более суток - консультацию закрываю..»
Вопросы: 1,5,3,4,5
Шаблон для ответа:
Вопросы: [5 цифр через запятую]
Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот диалог клиента с психологом, проанализируй его и ответь на вопросы в соответствии с описанными выше критериями и правилами:  {text}"""


async def process_dialog(dialogue, session, sem, api_base, model, out_dir, prompt_prefix, max_tokens, done_ids, pbar):
    link = dialogue.get("link") or "nolink"
    title = dialogue.get("title") or ""
    full_text = dialogue.get("text") or ""
    sid = extract_session_id(link)
    if sid in done_ids:
        pbar.update(1)
        return
    if not full_text.strip():
        pbar.update(1)
        return
    prompt = prompt_prefix.format(text=full_text)
    ans = await gen(session, sem, api_base, model, prompt, max_tokens=max_tokens,
                    stats_path="run_stats_quest.jsonl",
                    meta={"task": "quest", "sid": sid})
    out_path = os.path.join(out_dir, f"result_{sid}_quest.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump({link: {"title": title, "text": full_text, "annotated_dialogue": ans.strip()}},
                  fw, ensure_ascii=False, indent=2)
    pbar.update(1)


async def async_main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="1")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()

    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as session:
        for fp in args.files:
            if not os.path.exists(fp):
                print(f"Файл {fp} не найден, пропускаем...", file=sys.stderr)
                continue
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            dialogues = parse_dialogues(data)
            done_ids = get_done_ids_dialog(args.out)
            os.makedirs(args.out, exist_ok=True)
            pbar = tqdm(total=len(dialogues), desc="questions_dialogs")
            await asyncio.gather(*[
                process_dialog(d, session, sem, args.api, args.model, args.out,
                               PROMPT_TMPL, args.max_tokens, done_ids, pbar)
                for d in dialogues
            ])
            pbar.close()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
