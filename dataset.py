import warnings
import os
import pickle
import pandas as pd
from datasets import Dataset
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import login
from datasets import load_dataset
from conllu import parse_incr

class FT_Dataset:
    def __init__(self, EOS_TOKEN, split="train", shots=0, logger = None, test_mode=False, shuffle=False, prompt_type="ins"):
        login(token="hf_Key")
        print("test_mode: ", test_mode)
        assert shots in [0, 1, 3, 5, 10], "Shots should be one of 0, 1, 3, 5, 10"
        self.shots = shots

        self.EOS_TOKEN = "" if test_mode else EOS_TOKEN
        self.split = split
        self.logger = logger
        self.test_mode = test_mode
        self.prompt_type = prompt_type

        self.shuffle = shuffle

        print("WILL SHUFFLE: " + str(self.shuffle) + " =====================================")

        self.dataset_names = {
            # "dialect_train": "./data/madar_train_balanced.jsonl",
            # "dialect_test":  "./data/madar_test_balanced.jsonl",
            # "dialecttomsa_train": "./data/madar_train_balanced.jsonl",
            # "dialecttomsa_test":  "./data/madar_test_balanced.jsonl",
            # "msatodialect_train": "./data/madar_train_balanced.jsonl",
            # "msatodialect_test":  "./data/madar_test_balanced.jsonl",
            "dialect_train": "./data/madar_train_big_5_cities.jsonl",
            "dialect_test":  "./data/madar_test_big_5_cities.jsonl",
            "dialecttomsa_train": "./data/madar_train_big_5_cities.jsonl",
            "dialecttomsa_test":  "./data/madar_test_big_5_cities.jsonl",
            "msatodialect_train": "./data/madar_train_big_5_cities.jsonl",
            "msatodialect_test":  "./data/madar_test_big_5_cities.jsonl",
        }

        self.dataset_splits = {

            "dialect_train": "train",
            "dialect_test":  "train",
            "dialecttomsa_train": "train",
            "dialecttomsa_test":  "train",
            "msatodialect_train": "train",
            "msatodialect_test":  "train",
        }

        self.subset_names = {
        }

        self.prompt_func_map = {

            "dialect_train": self.format_prompt_dialect,
            "dialect_test": self.format_prompt_dialect,
            "dialecttomsa_train": self.format_prompt_d2m,
            "dialecttomsa_test":  self.format_prompt_d2m,
            "msatodialect_train": self.format_prompt_m2d,
            "msatodialect_test":  self.format_prompt_m2d,
        }

        # =============================================
        self.task_instructions = {
            "dialect": "",
            "dialecttomsa": "",
            "msatodialect": "",
        }

        self.task_instructions_ar = {
            "dialect": "",
            "dialecttomsa": "",
            "msatodialect": "",
        }
        # =============================================


        self.size = -1

        # self.city_mapping = {
        #     "ALE": {"en": "Aleppo",      "ar": "حلب"},
        #     "ALG": {"en": "Algiers",     "ar": "الجزائر"},
        #     "ALX": {"en": "Alexandria",  "ar": "الإسكندرية"},
        #     "AMM": {"en": "Amman",       "ar": "عمّان"},
        #     "ASW": {"en": "Aswan",       "ar": "أسوان"},
        #     "BAG": {"en": "Baghdad",     "ar": "بغداد"},
        #     "BAS": {"en": "Basra",       "ar": "البصرة"},
        #     "BEI": {"en": "Beirut",      "ar": "بيروت"},
        #     "BEN": {"en": "Benghazi",    "ar": "بنغازي"},
        #     "CAI": {"en": "Cairo",       "ar": "القاهرة"},
        #     "DAM": {"en": "Damascus",    "ar": "دمشق"},
        #     "DOH": {"en": "Doha",        "ar": "الدوحة"},
        #     "FES": {"en": "Fes",         "ar": "فاس"},
        #     "JED": {"en": "Jeddah",      "ar": "جدة"},
        #     "JER": {"en": "Jerusalem",   "ar": "القدس"},
        #     "KHA": {"en": "Khartoum",    "ar": "الخرطوم"},
        #     "MOS": {"en": "Mosul",       "ar": "الموصل"},
        #     "MUS": {"en": "Muscat",      "ar": "مسقط"},
        #     "RAB": {"en": "Rabat",       "ar": "الرباط"},
        #     "RIY": {"en": "Riyadh",      "ar": "الرياض"},
        #     "SAL": {"en": "Salt",        "ar": "السلط"},
        #     "SAN": {"en": "Sana'a",      "ar": "صنعاء"},
        #     "SFX": {"en": "Sfax",        "ar": "صفاقس"},
        #     "TRI": {"en": "Tripoli",     "ar": "طرابلس"},
        #     "TUN": {"en": "Tunis",       "ar": "تونس"},
        # }

        self.city_mapping = {
            "BEI": {"en": "Beirut",      "ar": "بيروت"},
            "CAI": {"en": "Cairo",       "ar": "القاهرة"},
            "DOH": {"en": "Doha",        "ar": "الدوحة"},
            "RAB": {"en": "Rabat",       "ar": "الرباط"},
            "TUN": {"en": "Tunis",       "ar": "تونس"},
        }

        # Build label lists for prompts
        self.DIALECT_CITY_LABELS_EN = [v["en"] for v in self.city_mapping.values()]
        self.DIALECT_CITY_LABELS_AR = [v["ar"] for v in self.city_mapping.values()]

        self.DIALECT_LABELS_STR_EN = ", ".join(self.DIALECT_CITY_LABELS_EN)
        self.DIALECT_LABELS_STR_AR = ", ".join(self.DIALECT_CITY_LABELS_AR)


        # ---------------------------------------------------------
        # Dialect Prompt Templates (no "النص:/الإجابة:" pattern)
        # Each template has ONE {} placeholder for the sentence.
        # Few-shot examples will be prepended separately.
        # ---------------------------------------------------------
        self.dialect_prompts = {

            # ===================== ENGLISH PROMPTS =====================
            "en_ins": (
                "You are a dialect classifier for Arabic sentences.\n"
                f"The possible dialect labels are: {self.DIALECT_LABELS_STR_EN}.\n\n"
                "Your task is to identify the dialect of the sentence below.\n"
                "You must answer with ONLY ONE CITY NAME from the list above.\n"
                "The final answer MUST be placed strictly inside <answer>...</answer>.\n"
                "Do NOT output any explanation, reasoning, or additional text.\n\n"
                "Stop after ONE single answer. Do NOT continue.\n\n"
                'Sentence: "{}"\n\n'
                "<answer>{}"
            ),

            "en_int": (
                "You are being interviewed as an Arabic dialect identification expert.\n"
                f"You must choose EXACTLY one city name from the following list:\n{self.DIALECT_LABELS_STR_EN}\n\n"
                "Your reply must contain ONLY the city name, placed inside <answer>...</answer>.\n"
                "Do NOT include any explanation.\n\n"
                "Stop after ONE single answer. Do NOT continue.\n\n"
                'Sentence: "{}"\n\n'
                "<answer>{}"
            ),

            "en_rp": (
                "You are an AI system specialized in identifying the dialect of Arabic sentences.\n"
                f"The valid city labels are: {self.DIALECT_LABELS_STR_EN}.\n\n"
                "Select the correct city name and output it ONLY inside <answer>...</answer>.\n"
                "No other text is allowed.\n\n"
                "Stop after ONE single answer. Do NOT continue.\n\n"
                'Sentence: "{}"\n\n'
                "<answer>{}"
            ),

            # ===================== ARABIC PROMPTS =====================
            "ar_ins": (
                "أنت مصنف لهجات عربية.\n"
                f"الخيارات الممكنة هي: {self.DIALECT_LABELS_STR_AR}.\n\n"
                "مهمتك هي تحديد اسم المدينة التي تمثل لهجة الجملة.\n"
                "الإجابة يجب أن تكون اسم مدينة واحد فقط داخل <answer>...</answer>.\n"
                "يُمنع كتابة أي شرح أو نص إضافي.\n"
                "بعد وضع الإجابة داخل الوسم، يجب إنهاء الرسالة فوراً.\n\n"
                "توقف بعد إجابة واحدة فقط. لا تتابع.\n\n"
                'الجملة: "{}"\n\n'
                "<answer>{}"
            ),

            "ar_int": (
                "يتم سؤالك بصفتك خبيرًا في تمييز اللهجات العربية.\n"
                f"يجب اختيار اسم مدينة واحد فقط من القائمة التالية: {self.DIALECT_LABELS_STR_AR}.\n"
                "ضع إجابتك داخل <answer>...</answer> فقط دون أي كلام إضافي.\n"
                "لا تكتب أي شيء بعد إغلاق الوسم.\n\n"
                "توقف بعد إجابة واحدة فقط. لا تتابع.\n\n"
                'الجملة: "{}"\n\n'
                "<answer>{}"
            ),

            "ar_rp": (
                "أنت نظام ذكاء اصطناعي متخصص في تحديد لهجة الجمل العربية.\n"
                f"المدن المحتملة هي: {self.DIALECT_LABELS_STR_AR}.\n\n"
                "أرجع اسم مدينة واحد فقط يمثل اللهجة، ويجب أن يكون داخل <answer>...</answer>.\n"
                "لا تكتب أي تفسير أو نص آخر بعد ذلك.\n\n"
                "توقف بعد إجابة واحدة فقط. لا تتابع.\n\n"
                'الجملة: "{}"\n\n'
                "<answer>{}"
            ),
        }

        # ==============================================================
        # Dialect  →  MSA  (d2m)
        # ==============================================================

        self.translation_prompts_d2m = {

            # -----------------------------------------------------
            # ENGLISH PROMPTS
            # -----------------------------------------------------
            "en_ins": (
                "Convert the following Arabic dialect sentence into Modern Standard Arabic (MSA).\n"
                "Inside <answer>...</answer>, return ONLY the final MSA sentence between double quotes.\n"
                "Do NOT add any extra text, explanation, or prefixes such as 'the correct text is'.\n"
                "Stop writing immediately after closing </answer>.\n\n"
                "Stop after ONE single answer. Do NOT continue.\n\n"
                'Sentence: "{}"\n\n'
                "<answer>{}"
            ),

            "en_int": (
                "You are being interviewed as a linguistic expert.\n"
                "Your task is to rewrite the following Arabic dialect sentence in MSA.\n"
                "Return ONLY the MSA sentence between quotes inside <answer>...</answer>.\n"
                "Do NOT add commentary or justification phrases.\n"
                "Stop writing immediately after </answer>.\n\n"
                "Stop after ONE single answer. Do NOT continue.\n\n"
                'Sentence: "{}"\n\n'
                "<answer>{}"
            ),

            "en_rp": (
                "You are an AI system specialized in Arabic normalization.\n"
                "Rewrite the text below into Modern Standard Arabic (MSA).\n"
                "Output ONLY the rewritten MSA sentence between quotes inside <answer>...</answer>.\n"
                "NEVER add prefixes or commentary such as 'here is the corrected version'.\n"
                "Stop writing immediately after </answer>.\n\n"
                "Stop after ONE single answer. Do NOT continue.\n\n"
                'Sentence: "{}"\n\n'
                "<answer>{}"
            ),


            # -----------------------------------------------------
            # ARABIC PROMPTS
            # -----------------------------------------------------
            "ar_ins": (
                "قم بإعادة صياغة الجملة التالية باللغة العربية الفصحى.\n"
                "يجب أن تكون الإجابة النهائية داخل <answer>...</answer> فقط.\n"
                "يمنع تمامًا كتابة أي مقدمات أو عبارات مثل: \"النص الصحيح هو\".\n"
                "ضع الجملة النهائية فقط بين علامتي تنصيص.\n"
                "توقف عن الكتابة مباشرة بعد إغلاق الوسم.\n\n"
                "توقف بعد إجابة واحدة فقط. لا تتابع.\n\n"
                'الجملة: "{}"\n\n'
                "<answer>{}"
            ),

            "ar_int": (
                "يتم تقييمك بصفاتك خبيرًا لغويًا.\n"
                "مهمتك هي تحويل الجملة التالية إلى اللغة العربية الفصحى.\n"
                "ضع الجملة النهائية فقط بين علامتي تنصيص داخل <answer>...</answer>.\n"
                "يمنع كتابة أي مقدمات أو إضافات مثل: \"النص الصحيح هو\".\n"
                "توقف فورًا بعد إغلاق الوسم.\n\n"
                "توقف بعد إجابة واحدة فقط. لا تتابع.\n\n"
                'الجملة: "{}"\n\n'
                "<answer>{}"
            ),

            "ar_rp": (
                "أنت نظام متخصص في تحويل اللهجات العربية إلى الفصحى.\n"
                "قم بإعادة كتابة الجملة التالية بالفصحى بدون أي شرح أو مقدمات.\n"
                "يجب أن تحتوي الوسوم <answer>...</answer> على الجملة النهائية فقط بين علامتي تنصيص.\n"
                "ويمنع استخدام عبارات مثل: \"النص الصحيح هو\".\n"
                "توقف عن الكتابة مباشرة بعد إغلاق الوسم.\n\n"
                "توقف بعد إجابة واحدة فقط. لا تتابع.\n\n"
                'الجملة: "{}"\n\n'
                "<answer>{}"
            ),
        }

        # ==============================================================
        # MSA  →  Dialect  (m2d)
        # ==============================================================

        self.translation_prompts_m2d = {
            # -----------------------------------------------------
            # ENGLISH PROMPTS
            # -----------------------------------------------------
            "en_ins": (
                "Rewrite the following text in the {} dialect.\n"
                "Inside <answer>...</answer>, return ONLY the rewritten sentence between double quotes.\n"
                "Do NOT add any extra text, explanation, or prefixes such as 'the corrected text is'.\n"
                "Stop writing immediately after closing </answer>.\n\n"
                "Stop after ONE single answer. Do NOT continue.\n\n"
                'Sentence: "{}"\n\n'
                "<answer>{}"
            ),

            "en_int": (
                "You are being interviewed as a dialect specialist.\n"
                "Your task is to rewrite the following sentence into the {} dialect.\n"
                "Return ONLY the rewritten sentence between quotes inside <answer>...</answer>.\n"
                "Do NOT add commentary or justification phrases. \n"
                "Stop writing immediately after </answer>.\n\n"
                "Stop after ONE single answer. Do NOT continue.\n\n"
                'Sentence: "{}"\n\n'
                "<answer>{}"
            ),

            "en_rp": (
                "You are an AI system specialized in Arabic dialect generation.\n"
                "Convert the text below into the {} dialect.\n"
                "Output ONLY the final dialect sentence between quotes inside <answer>...</answer>.\n"
                "NEVER add prefixes or commentary such as 'here is the rewritten version'.\n"
                "Stop writing immediately after </answer>.\n\n"
                "Stop after ONE single answer. Do NOT continue.\n\n"
                'Sentence: "{}"\n\n'
                "<answer>{}"
            ),


            # -----------------------------------------------------
            # ARABIC PROMPTS
            # -----------------------------------------------------
            "ar_ins": (
                "أعد كتابة الجملة التالية بلهجة: {}.\n"
                "يجب أن تكون الإجابة النهائية داخل <answer>...</answer> فقط.\n"
                "يمنع تمامًا كتابة أي مقدمات أو عبارات مثل: \"النص الصحيح هو\".\n"
                "ضع الجملة النهائية فقط بين علامتي تنصيص.\n"
                "توقف عن الكتابة مباشرة بعد إغلاق الوسم.\n\n"
                "توقف بعد إجابة واحدة فقط. لا تتابع.\n\n"
                'الجملة: "{}"\n\n'
                "<answer>{}"
            ),

            "ar_int": (
                "يتم تقييمك بصفتك خبيرًا في اللهجات العربية.\n"
                "قم بتحويل الجملة التالية إلى لهجة: {}.\n"
                "ضع الجملة النهائية فقط بين علامتي تنصيص داخل <answer>...</answer>.\n"
                "يمنع كتابة أي مقدمات أو إضافات مثل: \"النص الصحيح هو\".\n"
                "توقف فورًا بعد إغلاق الوسم.\n\n"
                "توقف بعد إجابة واحدة فقط. لا تتابع.\n\n"
                'الجملة: "{}"\n\n'
                "<answer>{}"
            ),

            "ar_rp": (
                "أنت نظام متخصص في تحويل اللغة العربية الفصحى إلى اللهجات.\n"
                "قم بإعادة كتابة الجملة التالية بلهجة: {} بدون أي شرح أو مقدمات.\n"
                "يجب أن تحتوي الوسوم <answer>...</answer> على الجملة النهائية فقط بين علامتي تنصيص.\n"
                "ويمنع استخدام عبارات مثل: \"النص الصحيح هو\".\n"
                "توقف عن الكتابة مباشرة بعد إغلاق الوسم.\n\n"
                "توقف بعد إجابة واحدة فقط. لا تتابع.\n\n"
                'الجملة: "{}"\n\n'
                "<answer>{}"
            ),
        }


    def get_size(self):
        assert self.size > 0, "Call get_dataset() first !!!"
        return self.size
    
    def format_prompt_dialect(self, data): 
        sents = data["dialect_sentence"]
        tags = data["dialect"]
        texts = []
        golds = []

        # ------------------- FEW SHOTS -------------------
        examples = ""
        if self.shots > 0:
            lang_key = "en" if self.lang == "en" else "ar"

            # build example section without <answer> and WITHOUT same structure
            ex_lines = []

            # header
            if lang_key == "en":
                ex_lines.append("Illustrative examples (do NOT follow this format in the answer):")
            else:
                ex_lines.append("أمثلة توضيحية (لا تحاول تقليد هيكل المثال في إجابتك):")

            max_shots = min(self.shots, len(sents))
            for i in range(max_shots):
                sent_i = sents[i]
                code_i = tags[i]
                city_i = self.city_mapping[code_i][lang_key]

                # DO NOT USE <answer> here!
                if lang_key == "en":
                    ex_lines.append(f'{i+1}. "{sent_i}" → {city_i}')
                else:
                    ex_lines.append(f'{i+1}. "{sent_i}" → {city_i}')

            # closing note
            if lang_key == "en":
                ex_lines.append("(End of examples)\n")
            else:
                ex_lines.append("(انتهت الأمثلة)\n")

            examples = "\n".join(ex_lines) + "\n\n"

            print("examples: ", examples)

        for inp, output in zip(sents, tags):
            output = self.city_mapping[output][self.lang]
            text = self.prompt_template.format(examples, inp, output+"</answer>" if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
            golds.append(output)

        return {"text": texts, "gold": golds}

    def format_prompt_d2m(self, data):
        dialect_inputs = data["dialect_sentence"]   # input text (dialect)
        msa_targets = data["msa"]      # gold MSA
        texts = []
        golds = []

        # ----------- FEW-SHOTS (safe format, no <answer>!) -----------
        examples = ""
        if self.shots > 0:
            lang_key = "en" if self.lang == "en" else "ar"
            ex_lines = []
            
            # Header
            if lang_key == "en":
                ex_lines.append("Illustrative examples (do NOT reproduce this style):")
            else:
                ex_lines.append("أمثلة توضيحية (لا تقلد الأسلوب في الإجابة):")

            max_shots = min(self.shots, len(dialect_inputs))
            for i in range(max_shots):
                di = dialect_inputs[i]
                mi = msa_targets[i]
                # NO <answer> here!
                ex_lines.append(f'{i+1}. "{di}" → "{mi}"')

            # Footer
            if lang_key == "en":
                ex_lines.append("(End of examples)\n")
            else:
                ex_lines.append("(انتهت الأمثلة)\n")

            examples = "\n".join(ex_lines) + "\n\n"

        # ----------- MAIN PROMPTS -----------    
        for inp, output in zip(dialect_inputs, msa_targets):
            # translation_prompts_d2m template has 2 placeholders: (examples, sentence)
            text = self.prompt_template.format(examples, inp, output+"</answer>" if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
            golds.append(output)

        return {"text": texts, "gold": golds}

    def format_prompt_m2d(self, data):
        msa_inputs = data["msa"]       # input text (MSA)
        dialect_targets = data["dialect_sentence"]   # gold dialect text
        dialect_codes = data["dialect"]            # "CAI", "DAM", ...
        texts = []
        golds = []

        # ----------- FEW-SHOTS -----------
        examples = ""
        if self.shots > 0:
            lang_key = "en" if self.lang == "en" else "ar"
            ex_lines = []

            # Header
            if lang_key == "en":
                ex_lines.append("Illustrative examples (do NOT copy this format):")
            else:
                ex_lines.append("أمثلة توضيحية (لا تحاول تقليد الأسلوب):")

            max_shots = min(self.shots, len(msa_inputs))
            for i in range(max_shots):
                mi = msa_inputs[i]
                di = dialect_targets[i]
                code_i = dialect_codes[i]
                city_i = self.city_mapping[code_i][lang_key]

                ex_lines.append(f'{i+1}. "{mi}" → ({city_i}) "{di}"')

            # Footer
            if lang_key == "en":
                ex_lines.append("(End of examples)\n")
            else:
                ex_lines.append("(انتهت الأمثلة)\n")

            examples = "\n".join(ex_lines) + "\n\n"

        # ----------- MAIN PROMPTS -----------
        for inp, code, output in zip(msa_inputs, dialect_codes, dialect_targets):
            lang_key = "en" if self.lang == "en" else "ar"

            # get city name based on dialect code
            city_name = self.city_mapping[code][lang_key]

            # template has 3 placeholders: (examples, city_name, sentence)
            text = self.prompt_template.format(examples, city_name, inp, output+"</answer>" if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
            golds.append(output)

        return {"text": texts, "gold": golds}


    def format_prompt_translation(self, data):
        sourceStrings = data["sourceString"]
        targetStrings = data["targetString"]
        texts = []

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(sourceStrings), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + sourceStrings[i] + "\n\n" + self.a_head + "<answer>" + targetStrings[i] + "</answer>\n\n"

        examples = ""
        shots_sourceStrings = [
            "تعليقات الحكومات على تقرير الفريق العامل المعني",
            "بمسألة إنشاء قضاء جنائي دولي",
            "٣ - قدمت استراليا في البيان الذي أدلت به في أثناء مناقشة هذا الموضوع في اللجنة السادسة في ٢٨ تشرين اﻷول/أكتوبر ١٩٩٢، تقييما للنهج العام الذي يتبعه الفريق العامل وأشارت الى أهمية العناصر التالية في ذلك النهج :",
            "ومن الجلي أن عبء العمل في المحكمة سيكون أيضا أشد محدودية، متى كانت الوﻻية التي تمارسها متفقة مع وﻻيات المحاكم الوطنية أكثر من كونها وﻻية خاصة.",
            "ويعكس هذا الموقف تفهما لعبء العمل المحدود الذي قد تواجهه المحكمة المرتآة، في سنوات عملها اﻷولى على اﻷقل، والتكاليف التي قد تتكبد نتيجة ﻹنشاء محكمة واﻹبقاء عليها كهيئة متفرغة تضم مجموعة كاملة من القضاة وهيكﻻ إداريا داعما."
        ]
        shots_targetStrings = [
            "COMMENTS OF GOVERNMENTS ON THE REPORT OF THE WORKING GROUP",
            "ON THE QUESTION OF AN INTERNATIONAL CRIMINAL JURISDICTION",
            "3. In its intervention during the debate on this issue in the Sixth Committee on 28 October 1992, Australia assessed the general approach of the Working Group and noted the importance of the following elements of that approach:",
            "he workload of a court would also clearly be more limited if it exercised concurrent jurisdiction with national courts rather than exclusive jurisdiction.",
            "This position reflects an understanding of the limited workload that a court would face, at least in its early years of operation, and the costs that would be incurred in establishing and maintaining a court on a full-time basis with a full complement of judges and a supporting administrative structure."
        ]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shots_sourceStrings[i] + "\n\n" + self.a_head + "<answer>" + shots_targetStrings[i] + "</answer>\n\n"

        for sourceString, targetString in zip(sourceStrings, targetStrings):
            text = self.prompt_template.format(examples, sourceString, targetString if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}

    
    def format_prompt_transliteration(self,data):
        EN = data["source"]
        AR = data["transliteration"]

        # examples = ""
        # if self.shots > 0:
        #     examples = self.e_head
        #     indices = np.random.choice(len(EN), self.shots, replace=False)
        #     for i in indices:
        #         examples += self.q_head + EN[i] + "\n\n" + self.a_head + "<answer>" + AR[i] + "</answer>\n\n"

        examples = ""
        shots_EN = [
            "Btgahzo el flat!!",
            "Enty ya benty msh btrodii 3la elbta3 da abdan",
            "2a5eraaan",
            "w stress sho3'lo",
            "enty 3amlah yom 7'ames w elnas 7trg3 mn sho3'lha w try7 w tgelk",
        ]
        shots_AR = [
            "بتجهزوا الفلت!!",
            "انتي يا بنتي مش بتردي على البتاع ده ابدا",
            "أخيران",
            "وسترس شغله",
            "انتي عملاه يوم خميس والناس حترجع من شغلها وتريح وتجي لك"
        ]
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shots_EN[i] + "\n\n" + self.a_head + "<answer>" + shots_AR[i] + "</answer>\n\n"
        
        texts = []
        for en, ar in zip(EN, AR):
            text = self.prompt_template.format(examples, en, ar if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts} 


    def construct_prompt(self, task, lang):
        if task == "dialect":
            # choose prompt template according to lang + prompt_type
            key = f"{lang}_{self.prompt_type}"
            assert key in self.dialect_prompts, f"Invalid prompt type: {self.prompt_type}"

            self.prompt_template = "{}" + self.dialect_prompts[key]

            return
        
        if task == "dialecttomsa":
            key = f"{lang}_{self.prompt_type}"
            self.prompt_template = "{}" + self.translation_prompts_d2m[key]
            return

        if task == "msatodialect":
            key = f"{lang}_{self.prompt_type}"
            self.prompt_template = "{}" + self.translation_prompts_m2d[key]
            return

        if lang == "en":
            self.prompt_template = "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            self.prompt_template += "Write a response that appropriately completes the request.\n"
            self.prompt_template += "Dont say anything except the answer. Give the final answer between answer tags: <answer>...</answer>.\n"
            self.prompt_template += "\n"
            self.prompt_template += "### Instruction:\n"
            self.prompt_template += f"{self.task_instructions[task]}\n"
            self.prompt_template += "\n"
            self.prompt_template += "{}"
            self.prompt_template += "\n"
            self.prompt_template += "-------------------\n" if self.shots>0 else ""
            self.prompt_template += f"### Question:\n"
            self.prompt_template += "{}"
            self.prompt_template += "\n\n"
            self.prompt_template += f"### Response:\n"
            self.prompt_template += "{}"

        elif lang == "ar":
            # self.prompt_template = "يوجد أدناه تعليمات تصف مهمة، مقترنة بإدخال يوفر سياقًا إضافيًا." + "\n"
            # self.prompt_template += "اكتب الرد الذي يكمل الطلب بشكل مناسب." + "\n"
            # self.prompt_template += "لا تقل أي شيء باستثناء الإجابة. أعط الإجابة النهائية بين علامات الإجابة: <answer>...</answer>.\n"
            # self.prompt_template += "\n"
            # self.prompt_template += ":تعليمات" + "###" + "\n"
            # self.prompt_template += f"{self.task_instructions_ar[task]}\n"
            # self.prompt_template += "\n"
            # self.prompt_template += "{}"
            # self.prompt_template += "\n"
            # self.prompt_template += "-------------------\n" if self.shots>0 else ""
            # self.prompt_template += ":سؤال" + "###" + "\n"
            # self.prompt_template += "{}"
            # self.prompt_template += "\n\n"
            # self.prompt_template += ":إجابة" + "###" + "\n"
            # self.prompt_template += "{}"

            self.prompt_template  = ""
            self.prompt_template += "يوجد أدناه تعليمات تصف مهمة لغوية.\n"
            self.prompt_template += "يمكنك التفكير داخل الوسم <think>...</think> فقط.\n"
            self.prompt_template += "بعد الانتهاء من التفكير، يجب أن تكتب الإجابة النهائية داخل وسم واحد فقط:\n"
            self.prompt_template += "<answer> ... </answer>\n"
            self.prompt_template += "ممنوع كتابة أي شيء خارج وسم <answer> بعد الانتهاء من التفكير.\n"
            self.prompt_template += "أي مخرجات خارج <answer> سيتم تجاهلها بالكامل.\n"
            self.prompt_template += "\n"

            self.prompt_template += ":تعليمات###\n"
            self.prompt_template += f"{self.task_instructions_ar[task]}\n"
            self.prompt_template += "\n"

            self.prompt_template += "الإجابة النهائية داخل <answer> يجب أن تكون JSON فقط بالشكل التالي:\n"
            self.prompt_template += "{{\n"
            self.prompt_template += '  "tokens": [\n'
            self.prompt_template += '    {{ "word": "الكلمة1", "label": "الوسم" }},\n'
            self.prompt_template += '    {{ "word": "الكلمة2", "label": "الوسم" }}\n'
            self.prompt_template += "  ]\n"
            self.prompt_template += "}}\n"
            self.prompt_template += "لا تكتب أي نص آخر.\n"
            self.prompt_template += "\n"

            # few-shot examples
            self.prompt_template += "{}\n"

            if self.shots > 0:
                self.prompt_template += "-------------------\n"

            self.prompt_template += ":سؤال###\n"
            self.prompt_template += "{}\n\n"

            self.prompt_template += ":إجابة###\n"
            self.prompt_template += "{}"


        else:
            if self.logger is not None:
                self.logger(lang + " not supported")
            exit()

        if self.logger is not None:
            self.logger("PROMPT:")
            self.logger(self.prompt_template)
            self.logger("\n\n")


    def get_dataset(self, task, lang="ar"):
        self.lang = lang
        print(self.lang, "==========================")

        # self.q_head =  "## Question:\n" if self.lang == "en" else (":سؤال" + "##" + "\n")
        # self.a_head = "## Response:\n" if self.lang == "en" else (":إجابة" + "##" + "\n")
        
        
        self.q_head = "Sentence:\n" if self.lang == "en" else "النص:\n"
        self.a_head = ""
        self.e_head = "EXAMPLES:\n" if self.lang == "en" else "أمثلة:\n"
        
        self.construct_prompt(task, lang)
        task_split = task + "_" + self.split

        if os.path.exists(self.dataset_names[task_split]) and self.dataset_names[task_split].endswith(".csv"):
            dataset = load_dataset("csv", data_files=self.dataset_names[task_split])["train"]

        elif os.path.exists(self.dataset_names[task_split]) and self.dataset_names[task_split].endswith(".tsv"):
            df = pd.read_csv(self.dataset_names[task_split], delimeter="\t")
            dataset = Dataset.from_pandas(df)["train"]

        elif os.path.exists(self.dataset_names[task_split]) and self.dataset_names[task_split].endswith(".pkl"):
            with open(self.dataset_names[task_split], 'rb') as pickle_file:
                arabic_docs=pickle.load(pickle_file)

            flat_data = []
            for url, sections in arabic_docs.items():
                for section_name, section_data in sections.items():
                    flat_data.append({
                        'input_text': section_data['document'],
                        'target_text': section_data['summary'],
                    })

            df = pd.DataFrame(flat_data)
            dataset = Dataset.from_pandas(df)

        elif os.path.exists(self.dataset_names[task_split]) and self.dataset_names[task_split].endswith(".conllu"):
            examples = self.read_conllu_file(self.dataset_names[task_split])
            dataset = Dataset.from_list(examples)

        elif os.path.exists(self.dataset_names[task_split]) and self.dataset_names[task_split].endswith(".jsonl"):
            dataset = load_dataset("json", data_files=self.dataset_names[task_split], download_mode="force_redownload", keep_in_memory=True, cache_dir=None)["train"]

        else:
            dataset_name = self.dataset_names[task_split]
            subset_name = self.subset_names[task_split]
            dataset = load_dataset(dataset_name, subset_name, split=self.dataset_splits[task_split], trust_remote_code=True)

            # save as csv
            # df = pd.DataFrame(dataset)
            # df.to_csv("./train.csv", index=False)

        self.size = dataset.num_rows
        dataset = dataset.map(self.prompt_func_map[task_split], batched = True)
        
        if self.split == "train" and self.shuffle:
            dataset = dataset.shuffle(seed=42)

        if self.logger is not None:
            self.logger("\n\n")
            self.logger("DATASET SUMMARY:")
            self.logger(str(dataset))
            self.logger("\n\n")

            self.logger("EXAMPLE DATA INSTANCE:")
            self.logger(dataset["text"][-1])
            self.logger("\n\n")
        else:
            print("\n\n")
            print(task)
            print("DATASET SUMMARY")
            print(str(dataset))
            print("\n\n")

            print("EXAMPLE DATA INSTANCE:")
            print(dataset["text"][0])
            print()
            print("\n\n") 
            
            print("Length:", len(dataset["text"]))
            print("\n")

        return dataset


if __name__ == "__main__":
    # FT_Dataset("", split="test", shots=5).get_dataset("sentiment", "ar")
    # FT_Dataset("", split="train", shots=5).get_dataset("pos_tagging", "ar")
    FT_Dataset("", split="test", shots=5).get_dataset("summarization", "en")
    # FT_Dataset("", split="test", shots=5).get_dataset("translation", "ar")
    # FT_Dataset("", split="train", shots=5).get_dataset("paraphrasing", "en")
    # FT_Dataset("", split="test", shots=3).get_dataset("transliteration", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("sqs", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("stance", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("claim", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("wsd", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("GQA", "en")
    # FT_Dataset("", split="test", shots=5).get_dataset("sarcasm", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("dialect", "ar")
    # FT_Dataset("", split="test", shots=5).get_dataset("hate", "ar")
    # FT_Dataset("", split="test", shots=3).get_dataset("offensive", "en")