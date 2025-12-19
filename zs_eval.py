import os
import argparse
# from model import FT_Models

import sacrebleu
from rouge import Rouge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sacrebleu.metrics import BLEU, CHRF, TER
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

from collections import defaultdict

import re
import json
from utils import Logger
from evaluate import load

class Eval:
    def __init__(self, task, model_name="Q1.5B", prompt_lang="ar", preds_folder="./zs_preds5", prompt_type="ins"):
        self.task = task
        self.model_name = model_name
        self.prompt_lang = prompt_lang
        self.preds_folder = preds_folder
        self.prompt_type = prompt_type

        self.load_model()

        self.preds_file_path = os.path.join(self.preds_folder, "_".join([self.model_name, self.task, self.prompt_lang, self.prompt_type]))

        self.task_eval_map = {
            "sentiment": "classification",
            "pos_tagging": "multiclass_classification",
            "irab": "multiclass_classification_irab",
            "paraphrase_detection": "classification",
            "claim": "classification",
            "stance": "classification",
            "wsd": "classification",
            "paraphrasing": "bleu",
            "transliteration": "bleu",
            "translation": "bleu",
            "summarization": "rouge",
            "sarcasm": "classification",
            "dialect": "classification",
            "dialecttomsa": "translation",
            "msatodialect": "translation",
            "hate": "classification",
            "offensive": "classification",
            "sqs": "classification",
            "GQA": "squad"
        }

        self.eval_func_map = {
            #"classification": self.classification,
            "classification": self.score_classification_folder,
            "translation": self.score_translation_folder,
            "bleu": self.bleu,
            "rouge": self.rouge,
            "multiclass_classification": self.multiclass_classification,
            "multiclass_classification_irab": self.multiclass_classification_irab,
            "squad": self.squad
        }

        self.separator = "================================================================================="

    def evaluate(self):
        print("\n", self.task, self.prompt_lang,  '+++++++++++++++++')
        return self.eval_func_map[self.task_eval_map[self.task]]()
    

    def load_model(self):
        # self.tokenizer = FT_Models(self.model_name).get_tokenizer("R1-Q1.5B")
        self.eos_token = "｜end▁of▁sentence｜>" #self.tokenizer.eos_token


    def extract_json_from_answer(self, text):
        m = re.search(r"<answer>(.*?)</answer>", text, re.S)
        if not m:
            return None

        segment = m.group(1).strip()
        try:
            return json.loads(segment)
        except:
            return None


    def convert_irab_text_to_json(self, text):
        tokens = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "end▁of▁sentence" in line:
                continue

            if ":" not in line:
                continue

            parts = line.split(":")
            word = parts[0].strip()
            label = ":".join(parts[1:]).strip()

            tokens.append({"word": word, "label": label})
        
        return {"tokens": tokens}



    def get_preds(self):
        preds_folder = "_".join([self.model_name, self.task, self.prompt_lang, self.prompt_type])
        preds_dir = os.path.join(self.preds_folder, preds_folder)

        txt_files = os.listdir(preds_dir)
        if "scores.txt" in txt_files:
            txt_files.remove("scores.txt")
        txt_files = sorted(txt_files, key=lambda x: int(x.split('.')[0]))

        self.preds = []
        self.answers = []

        for i in range(len(txt_files)):
            with open(os.path.join(preds_dir, txt_files[i])) as pred_file:
                pred = pred_file.readlines()

            answer_bounds = []
            for i, p in enumerate(pred):
                if p in [self.separator, self.separator + "\n"]:
                    answer_bounds.append(i)

            answer = " ".join(pred[answer_bounds[0]+1: answer_bounds[1]])
            self.answers.append(answer.replace("\n", ""))

            pred = " ".join(pred[answer_bounds[1]+1:]).replace("\n", "")

            # Look for </think> first to extract only what comes after it
            think_match = re.search(r"</think>(.*)", pred, re.DOTALL)
            if think_match:
                pred_after_think = think_match.group(1).strip()
            else:
                pred_after_think = pred

            # Search for answer in the extracted portion
            answer_match = re.search(r"<answer>(.*?)</answer>", pred_after_think, re.DOTALL)
            if answer_match:
                self.preds.append(answer_match.group(1).strip())
            else:
                self.preds.append("<none>")

    def get_multiclass_preds(self):
        preds_folder = "_".join([self.model_name, self.task, self.prompt_lang, self.prompt_type])
        preds_dir = os.path.join(self.preds_folder, preds_folder)

        txt_files = os.listdir(preds_dir)
        if "scores.txt" in txt_files:
            txt_files.remove("scores.txt")
        txt_files = sorted(txt_files, key=lambda x: int(x.split('.')[0]))

        self.preds = []
        self.answers = []

        for i in range(len(txt_files)):
            with open(os.path.join(preds_dir, txt_files[i])) as pred_file:
                pred = pred_file.readlines()

            answer_bounds = []
            for i, p in enumerate(pred):
                if p in [self.separator, self.separator + "\n"]:
                    answer_bounds.append(i)

            answer = " ".join(pred[answer_bounds[0]+1: answer_bounds[1]])
            self.answers.append([answer.replace("\n", "")])

            pred = " ".join(pred[answer_bounds[1]+1:]).replace("\n", "")

            # Look for </think> first to extract only what comes after it
            think_match = re.search(r"</think>(.*)", pred, re.DOTALL)
            if think_match:
                pred_after_think = think_match.group(1).strip()
            else:
                pred_after_think = pred

            # Search for answer in the extracted portion
            answer_match = re.search(r"<answer>(.*?)</answer>", pred_after_think, re.DOTALL)
            if answer_match:
                self.preds.append([answer_match.group(1).strip()])
            else:
                self.preds.append(["<none>"])

    def get_multiclass_preds_irab(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                content = content.replace("<｜end▁of▁sentence｜>", "").replace("<|end_of_sentence|>", "").replace(self.eos_token, "")
        except Exception as e:
            self.log_eval_error(filepath, f"File read error: {e}")
            return [], []

        try:
            parts = content.split(self.separator)
            if len(parts) < 3:
                raise ValueError("Separator not found or incomplete sections.")

            gt_block = parts[1].strip()
            pred_block = parts[2].strip()

        except Exception as e:
            self.log_eval_error(filepath, f"Splitting sections failed: {e}")
            return [], []

        # ---------------------------------------------------
        #                PARSE GROUND TRUTH
        # ---------------------------------------------------
        try:
            if "{" in gt_block:
                try:
                    gt_json = json.loads(gt_block)
                except:
                    # maybe JSON is broken → log error
                    raise ValueError("GT JSON is invalid")

                # if isinstance(gt_json, list):
                #     gt_tokens = gt_json
                # else:
                gt_tokens = gt_json.get("tokens", [])
            else:
                # raw "word:label"
                gt_json = self.convert_irab_text_to_json(gt_block)
                gt_tokens = gt_json["tokens"]

            gt_labels = [tok["label"] for tok in gt_tokens]

        except Exception as e:
            self.log_eval_error(filepath, f"Ground truth parsing failed: {e}")
            # cannot guess true length → return empty and let caller skip
            return [], []

        # ---------------------------------------------------
        #                PARSE PREDICTION
        # ---------------------------------------------------
        try:
            pred_json = self.extract_json_from_answer(pred_block)

            if pred_json is None:
                pred_labels = ["INVALID"] * len(gt_labels)
            else:
                # if isinstance(pred_json, list):
                #     pred_tokens = pred_json
                # else:
                pred_tokens = pred_json.get("tokens", [])

                pred_labels = [tok.get("label", "INVALID") for tok in pred_tokens]

        except Exception as e:
            self.log_eval_error(filepath, f"Prediction parsing failed: {e}")
            pred_labels = ["INVALID"] * len(gt_labels)

        # ---------------------------------------------------
        #         FIX LENGTH (pad or truncate)
        # ---------------------------------------------------
        try:
            if len(pred_labels) < len(gt_labels):
                pred_labels += ["INVALID"] * (len(gt_labels) - len(pred_labels))
            else:
                pred_labels = pred_labels[:len(gt_labels)]
        except Exception as e:
            self.log_eval_error(filepath, f"Padding/truncation failed: {e}")
            pred_labels = ["INVALID"] * len(gt_labels)

        return gt_labels, pred_labels



    def squad(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        predictions = []
        references = []

        for i in range(len(self.preds)):
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.eos_token, "").replace("<｜end▁of▁sentence｜>", "")

            start = self.answers[i].find("[")
            end = self.answers[i].rfind("]")
            self.answers[i] = self.answers[i][start+1:end].replace('"', "")

            predictions.append({"id": str(i), "prediction_text": self.preds[i]})
            references.append({"id": str(i), "answers": {"text": [self.answers[i]], "answer_start": [0]}})

        return self.calculate_squad(predictions, references)


    def calculate_squad(self, preds, answers):
        squad_metric = load("squad")

        results = squad_metric.compute(predictions=preds, references=answers)

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"F1 Score: {str(results['f1'])}")

        return results['f1']


    def multiclass_classification(self):
        class_list = ['NOUN', 'PUNCT', 'ADP', 'NUM', 'SYM', 'SCONJ', 'ADJ', 'PART', 'DET', 'CCONJ', 'PROPN', 'PRON', 'X', 'ADV', 'INTJ', 'VERB', 'AUX']

        self.get_multiclass_preds()
        self.answers = self.answers[:len(self.preds)]

        def extract_classes(texts, class_list):
            texts = texts[0].split(" ")
            extracted = []
            for i, text in enumerate(texts):
                text = text.replace(self.eos_token, "").replace("<|end_of_sentence|>", "")
                for cl in class_list:
                    if cl in text:
                        extracted.append(cl)
                        break
            
            return extracted
        
        for i in range(len(self.preds)):
            self.preds[i] = extract_classes(self.preds[i], class_list)
            self.answers[i] = extract_classes(self.answers[i], class_list)


        final_answers, final_preds = [], []
        for i in range(len(self.preds)):
            if len(self.preds[i])>len((self.answers[i])):
                for j in range(len(self.answers[i])):
                    final_preds.append(self.preds[i][j])
                    final_answers.append(self.answers[i][j])
            
            else:
                for j in range(len(self.preds[i])):
                    final_preds.append(self.preds[i][j])
                    final_answers.append(self.answers[i][j])

                for j in range(len(self.preds[i]), len(self.answers[i])):
                    final_preds.append("DUMMY")
                    final_answers.append(self.answers[i][j])

        return self.calculate_F1(final_preds, final_answers)


    def multiclass_classification_irab(self):
        final_answers = []
        final_preds   = []
        preds_folder = "_".join([self.model_name, self.task, self.prompt_lang, self.prompt_type])
        preds_dir = os.path.join(self.preds_folder, preds_folder)

        if os.path.exists(os.path.join(preds_dir, "scores.txt")):
            os.remove(os.path.join(preds_dir, "scores.txt"))

        files = sorted(os.listdir(preds_dir), key=lambda x: int(x.replace(".txt","")))

        for fname in files:
            if not fname.endswith(".txt"):
                continue

            path = os.path.join(preds_dir, fname)

            gt, pred = self.get_multiclass_preds_irab(path)

            # If gt is empty → skip AND log
            if len(gt) == 0:
                self.log_eval_error(path, "Skipped due to empty GT (parsing failed).")
                continue

            final_answers.extend(gt)
            final_preds.extend(pred)

        # compute using your original F1 function
        results = self.calculate_F1(final_preds, final_answers)
        return results

    def log_eval_error(self, filepath, message):
        with open("irab_eval_errors.log", "a", encoding="utf-8") as f:
            f.write(f"[FILE]: {filepath}\n")
            f.write(f"[ERROR]: {message}\n")
            f.write("-----------------------------------------------------\n")


    def classification(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        def extract_first_digit(text):
            match = re.search(r"\d", text)
            return match.group(0) if match else text 

        for i in range(len(self.preds)):
            self.preds[i] = extract_first_digit(self.preds[i].replace("\n", "").replace(" ", "").strip())
            self.answers[i] = extract_first_digit(self.answers[i].replace("\n", "").replace(self.eos_token, "").replace("<｜end▁of▁sentence｜>", ""))

        # for i in zip(self.preds, self.answers):
        #     print(i)

        return self.calculate_F1(self.preds, self.answers)

    def bleu(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.eos_token, "").replace("<｜end▁of▁sentence｜>", "")

        return self.calculate_bleu(self.preds, self.answers)

    def rouge(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        self.preds = self.preds[9921:]
        self.answers = self.answers[9921:]

        for i in range(len(self.preds)):
            self.preds[i] = self.preds[i].replace("\n", "")
            self.answers[i] = self.answers[i].replace("\n", "").replace(self.eos_token, "").replace("<｜end▁of▁sentence｜>", "")

        return self.calculate_rouge(self.preds, self.answers)

    def calculate_F1(self, preds, answers):
        accuracy = accuracy_score(preds, answers)
        precision = precision_score(preds, answers, average='macro')
        recall = recall_score(preds, answers, average='macro')
        f1 = f1_score(preds, answers, average='macro')

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"Accuracy: {accuracy}")
        logger(f"Precision: {precision}")
        logger(f"Recall: {recall}")
        logger(f"F1 Score: {f1}")

        return accuracy, precision, recall, f1

    def calculate_bleu(self, preds, answers):
        bleu = sacrebleu.BLEU(effective_order=True)
        sentence_bleu_scores = [bleu.sentence_score(candidate, [reference]).score for reference, candidate in zip(answers, preds)]
        corpus_bleu_score = bleu.corpus_score(preds, [answers]).score
        avg_sentence_bleu_score = sum(sentence_bleu_scores) / len(sentence_bleu_scores) if sentence_bleu_scores else 0

        logger = Logger(os.path.join(self.preds_file_path, "scores.txt"))
        logger(f"Average Sentence BLEU score: {avg_sentence_bleu_score:.4f}")
        logger(f"Corpus BLEU score: {corpus_bleu_score:.4f}")

        return {
            "average_sentence_bleu": avg_sentence_bleu_score,
            "corpus_bleu": corpus_bleu_score
        }

    def calculate_rouge(self, preds, answers):
        rouge = Rouge()
        abstractive_rouge_1_scores, abstractive_rouge_2_scores, abstractive_rouge_l_scores = [], [], []
        for g_text, t_text in zip(preds, answers):
            try:
                scores = rouge.get_scores(g_text, t_text)[0]
                abstractive_rouge_1_scores.append(scores['rouge-1']['f'])
                abstractive_rouge_2_scores.append(scores['rouge-2']['f'])
                abstractive_rouge_l_scores.append(scores['rouge-l']['f'])
            except Exception as e:
                scores = rouge.get_scores(g_text[:1000], t_text[:1000])[0]
                abstractive_rouge_1_scores.append(scores['rouge-1']['f'])
                abstractive_rouge_2_scores.append(scores['rouge-2']['f'])
                abstractive_rouge_l_scores.append(scores['rouge-l']['f'])

        avg_abstractive_rouge_1 = sum(abstractive_rouge_1_scores) / len(abstractive_rouge_1_scores) if abstractive_rouge_1_scores else 0
        avg_abstractive_rouge_2 = sum(abstractive_rouge_2_scores) / len(abstractive_rouge_2_scores) if abstractive_rouge_2_scores else 0
        avg_abstractive_rouge_l = sum(abstractive_rouge_l_scores) / len(abstractive_rouge_l_scores) if abstractive_rouge_l_scores else 0

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(f"ROUGE-1: {avg_abstractive_rouge_1}")
        logger(f"ROUGE-2: {avg_abstractive_rouge_2}")
        logger(f"ROUGE-L: {avg_abstractive_rouge_l}")

        return avg_abstractive_rouge_1, avg_abstractive_rouge_2, avg_abstractive_rouge_l
    



    def per_label_accuracy(self, y_true, y_pred):
        total = defaultdict(int)
        correct = defaultdict(int)

        for t, p in zip(y_true, y_pred):
            total[t] += 1
            if t == p:
                correct[t] += 1

        acc = {lbl: correct[lbl] / total[lbl] for lbl in total}
        return acc

    def normalize_label(self, s):
        if s is None:
            return "<none>"
        # remove common special tokens / whitespace noise
        s = s.replace("<|im_end|>", "").replace("<|end_of_sentence|>", "").replace("<｜end▁of▁sentence｜>", "")
        s = s.strip()
        # keep only first line (GT often is a single line label)
        s = s.splitlines()[0].strip() if s else "<none>"
        return s

    def extract_pred_label(self, file_text):
        # Prefer content AFTER </think> if present
        m_think = re.search(r"</think>(.*)", file_text, flags=re.DOTALL)
        text = m_think.group(1) if m_think else file_text

        # Get ALL answer tags, then pick the LAST one (the model output)
        answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
        if not answers:
            return "<none>"

        return self.normalize_label(answers[-1])

    def extract_gt_label(self, file_text):
        # If there is a separator, take the block AFTER the first separator as GT
        parts = file_text.split(self.separator)
        if len(parts) >= 2:
            gt_block = parts[1]
            return self.normalize_label(gt_block)

        # fallback: try to find a standalone label at the very end
        tail = file_text.strip().splitlines()[-1] if file_text.strip() else "<none>"
        return self.normalize_label(tail)

    def score_classification_folder(self):
        files = [f for f in os.listdir(self.preds_file_path) if f.endswith(".txt")]
        
        if "scores.txt" in files:
            files.remove("scores.txt")
        
        files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))  # 1.txt, 2.txt, ...

        y_true, y_pred = [], []
        bad = 0

        for fn in files:
            path = os.path.join(self.preds_file_path, fn)
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()

            pred = self.extract_pred_label(txt)
            gt   = self.extract_gt_label(txt)

            if gt == "<none>":
                bad += 1
                continue

            y_true.append(gt)
            y_pred.append(pred)
        
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        result = {
            "n_scored": len(y_true),
            "n_skipped_no_gt": bad,
            "accuracy": float(f"{acc:.6f}"),
            "precision_macro": float(f"{p:.6f}"),
            "recall_macro": float(f"{r:.6f}"),
            "f1_macro": float(f"{f1:.6f}"),
        }

        logger = Logger(os.path.join(self.preds_file_path, f"scores.txt"))
        logger(result)
        
        labels = sorted(set(y_true))  # only labels that appear in GT

        logger("\n=== Per-Dialect Scores ===")
        logger(
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                digits=4,          # high precision
                zero_division=0
            )
        )

        per_acc = self.per_label_accuracy(y_true, y_pred)

        logger("\n=== Per-Dialect Accuracy ===")
        for lbl, a in sorted(per_acc.items()):
            logger(f"{lbl:>10s} : {a:.4f}")

        return result



    def score_translation_rouge(self, preds, refs):
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

        r1, r2, rl = [], [], []

        for p, r in zip(preds, refs):
            scores = scorer.score(r, p)
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rl.append(scores["rougeL"].fmeasure)

        return {
            "rouge1_f": sum(r1) / len(r1) if r1 else 0.0,
            "rouge2_f": sum(r2) / len(r2) if r2 else 0.0,
            "rougeL_f": sum(rl) / len(rl) if rl else 0.0,
        }

    def score_translation_meteor(self, preds, refs):
        scores = []
        for p, r in zip(preds, refs):
            scores.append(meteor_score([r.split()], p.split()))
        return sum(scores) / len(scores) if scores else 0.0

    def normalize_text(self, s: str) -> str:
        if s is None:
            return ""
        s = s.replace("<|im_end|>", "").replace("<|end_of_sentence|>", "").replace("<｜end▁of▁sentence｜>", "")
        s = s.replace(self.eos_token, "")
        s = s.strip()
        # collapse whitespace
        s = re.sub(r"\s+", " ", s)
        return s

    def extract_pred_translation(self, file_text: str) -> str:
        # Only look BEFORE separator (prediction is above it)
        before_sep = file_text.split(self.separator)[0]

        # take LAST <answer>...</answer>
        answers = re.findall(r"<answer>(.*?)</answer>", before_sep, flags=re.DOTALL)
        if not answers:
            return ""
        return self.normalize_text(answers[-1])

    def extract_gt_translation(self, file_text: str) -> str:
        parts = file_text.split(self.separator)
        if len(parts) >= 2:
            gt_block = parts[1]
            return self.normalize_text(gt_block)

        # fallback: last line
        tail = file_text.strip().splitlines()[-1] if file_text.strip() else ""
        return self.normalize_text(tail)


    def score_translation_folder(self):
        files = [f for f in os.listdir(self.preds_file_path) if f.endswith(".txt")]
        if "scores.txt" in files:
            files.remove("scores.txt")
        files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

        preds, refs = [], []
        skipped = 0

        for fn in files:
            path = os.path.join(self.preds_file_path, fn)
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()

            pred = self.extract_pred_translation(txt)
            ref  = self.extract_gt_translation(txt)

            if not ref:
                skipped += 1
                continue

            preds.append(pred)
            refs.append(ref)
            
        
        bleu = BLEU(effective_order=True)
        chrf = CHRF(word_order=2)   # good default for MT
        ter  = TER()

        # sentence-level BLEU (average) + corpus metrics
        sent_bleu = [bleu.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
        avg_sent_bleu = sum(sent_bleu) / len(sent_bleu) if sent_bleu else 0.0

        corpus_bleu = bleu.corpus_score(preds, [refs]).score
        corpus_chrf = chrf.corpus_score(preds, [refs]).score
        corpus_ter  = ter.corpus_score(preds, [refs]).score

        meteor = self.score_translation_meteor(preds, refs)
        rouge_scores = self.score_translation_rouge(preds, refs)

        result = {
            "n_scored": len(refs),
            "n_skipped_no_ref": skipped,
            "bleu": float(f"{corpus_bleu:.4f}"),
            "avg_sentence_bleu": float(f"{avg_sent_bleu:.4f}"),
            "chrf": float(f"{corpus_chrf:.4f}"),
            "ter": float(f"{corpus_ter:.4f}"),
            "meteor": float(f"{meteor:.4f}"),
            "rouge1_f": float(f"{rouge_scores['rouge1_f']:.4f}"),
            "rouge2_f": float(f"{rouge_scores['rouge2_f']:.4f}"),
            "rougeL_f": float(f"{rouge_scores['rougeL_f']:.4f}"),
        }

        logger = Logger(os.path.join(self.preds_file_path, "scores.txt"))
        logger(result)

        print("\n=== Translation Scores ===")
        for k, v in result.items():
            print(f"{k:>15s}: {v}")

        return result





if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',dest='model')
    parser.add_argument('--prompt_lang',dest='prompt_lang', default='ar', help='ar, en')
    parser.add_argument('--task',dest='task', default='sentiment')
    parser.add_argument('--preds_folder',dest='preds_folder', default='./zs_preds')
    parser.add_argument('--prompt_type', dest='prompt_type', default='ins')
    args=parser.parse_args()

    # assert args.model in ["Q1.5B", "Q7B", "Q14B"], "Invalid model!"
    assert args.prompt_lang in ["en", "ar"], "Only 'en' and 'ar' languages supported!"

    e = Eval(args.task, args.model, args.prompt_lang, args.preds_folder, prompt_type=args.prompt_type)
    e.evaluate()