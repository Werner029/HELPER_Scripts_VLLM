import argparse, asyncio, json, os, sys
import aiohttp
from tqdm import tqdm

from vllm_common import gen, get_done_ids_dialog, extract_session_id, parse_dialogues, split_turns

PROMPT_TMPL = """Ты — опытный психолог, специализирующийся на анализе качества
ответов психологов в консультативных диалогах. Тебе подаётся сообщение психолога(по одному), к которому обратился клиент, а далее у них строится диалог. Твоя задача — оценить сообщения психолога по
критериям. В процессе разметки диалога тебе предстоит дать оценку каждому сообщению психолога по следующим критериям: эмпатичность сообщения (от 1 до 5), этичность сообщения (от 1 до 5),
продуктивность вопроса (от 1 до 5, 6 - сообщение не является вопросом), полезность рекомендации (от 1 до 5, 6 - сообщение не содержит рекомендацию). Итоговой разметкой должна быть строка из 4 цифр без пробелов, где для каждого сообщения психолога 4 цифры.
Не забывай ставить 6, если сообщение не является рекомендацией или сообщением, а также старайся корректно оценить, используя четкий анализ ответа (оценка может быть любой от 1 до 5). Не пропускай никакие сообщения психолога, они все помечены как «Психолог:»
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
Оценка: 4466
Шаблон для ответа:
Оценка: [4 цифры] 
Теперь проанализируй диалог ниже и выполни оценку сообщений психолога (длина должна быть
равна 4_. Если цифр больше/меньше — исправь ошибку. Ответом является единая строка из цифр без
пробелов)
Не добавляй никаких пояснений, примечаний или блоков <think>.
Вот сообщение психолога, проанализируй его и выполни оценку в соответствии с описанными выше критериями и правилами:  {text}"""

async def process_dialog(dialogue, session, sem, api_base, model, out_dir, prompt_prefix, max_tokens, done_ids, pbar):
    link = dialogue.get("link") or "nolink"
    title = dialogue.get("title") or ""
    full_text = dialogue.get("text") or ""
    sid = extract_session_id(link)
    if sid in done_ids:
        pbar.update(1)
        return
    turns = split_turns(full_text)
    psych_turns = [t for t in turns if t and t.startswith("Психолог:")]
    results = await asyncio.gather(*[
        gen(session, sem, api_base, model, prompt_prefix.format(text=t), max_tokens=max_tokens,
            stats_path="run_stats_psych.jsonl",
            meta={"task": "psych", "sid": sid})
        for t in psych_turns
    ], return_exceptions=True)
    annotated = ["".join(ch for ch in (r if isinstance(r, str) else "") if ch.isdigit()) for r in results]
    out_path = os.path.join(out_dir, f"result_{sid}_psych.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump({link: {"title": title, "text": full_text, "annotated_dialogue": "".join(annotated)}},
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
            pbar = tqdm(total=len(dialogues), desc="psych_dialogs")
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
