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
    def __init__(self, EOS_TOKEN, split="train", shots=0, logger = None, test_mode=False, shuffle=False):
        login(token="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")

        assert shots in [0, 1, 3, 5, 10], "Shots should be one of 0, 1, 3, 5, 10"
        self.shots = shots

        self.EOS_TOKEN = "" if test_mode else EOS_TOKEN
        self.split = split
        self.logger = logger
        self.test_mode = test_mode

        self.shuffle = shuffle

        print("WILL SHUFFLE: " + str(self.shuffle) + " =====================================")

        self.dataset_names = {
            "dialect_train": "./data/madar_train_balanced.jsonl",
            "dialect_test":  "./data/madar_test_balanced.jsonl",


            "sentiment_train":"ajgt_twitter_ar",
            "sentiment_test":"ajgt_twitter_ar",

            "pos_tagging_train":"universal_dependencies",
            "pos_tagging_test":"universal_dependencies",

            "irab_train":"./data/ar_padt-ud-train.conllu",
            "irab_val":"./data/ar_padt-ud-dev.conllu",
            "irab_test":"./data/ar_padt-ud-test.conllu",

            "summarization_train":"./data/sum_train.csv",
            "summarization_test":"./data/sum_test.csv",

            "translation_train":"./data/translation_train.csv",
            "translation_test":"./data/translation_test.csv",

            "paraphrasing_train": "aishaalansari/paraphrase" ,
            "paraphrasing_test": "aishaalansari/Paraphrasing",

            "transliteration_train": "./data/transliteration_train.csv",
            "transliteration_test": "./data/transliteration_test.csv",

            "sqs_train": "./data/sqs_train.csv",
            "sqs_test": "./data/sqs_test.csv",

            "stance_train": "./data/stance_train.csv",
            "stance_test": "./data/stance_test.csv",

            "claim_train": "./data/claim_train.csv",
            "claim_test": "./data/claim_test.csv",

            "wsd_train": "./data/wsd_train.csv",
            "wsd_test": "./data/wsd_test.csv",

            # "mcq_train":"aishaalansari/CIDAR100",
            # "mcq_test":"aishaalansari/CIDAR100",

            "GQA_train": "asas-ai/tydiqa-goldp-ar",
            "GQA_test": "asas-ai/tydiqa-goldp-ar",

            # "diacratization_train":"arbml/tashkeelav2",
            # "diacratization_test":"arbml/tashkeelav2",

            "sarcasm_train": "./data/sarc_dab_train.csv",
            "sarcasm_test": "./data/sarc_dab_test.csv",

            # "dialect_train": "./data/sarc_dab_train.csv",
            # "dialect_test":  "./data/sarc_dab_test.csv",

            "hate_train": "./data/off_hs_train.csv",
            "hate_test": "./data/off_hs_test.csv",

            "offensive_train": "./data/off_hs_train.csv",
            "offensive_test": "./data/off_hs_test.csv",
        }

        self.dataset_splits = {

            "dialect_train": "train",
            "dialect_test":  "train",

            "sentiment_train":"train[:1440]",
            "sentiment_test":"train[1440:]",

            "pos_tagging_train":"train",
            "pos_tagging_test":"test",

            "irab_train":"train",
            "irab_val":"val",
            "irab_test":"test",

            "summarization_train":"train",
            "summarization_test":"train",

            "translation_train":"train",
            "translation_test":"test",

            "paraphrasing_train": "train",
            "paraphrasing_test": "train",

            "transliteration_train": "train",
            "transliteration_test": "test",

            "sqs_train":"train",
            "sqs_test":"test",

            "claim_train":"train",
            "claim_test":"test",

            "stance_train":"train",
            "stance_test":"test",

            # "mcq_train":"train",
            # "mcq_test":"test",

            "GQA_train": "train",
            "GQA_test": "validation",

            # "diacratization_train":"train",
            # "diacratization_test":"test",
        }

        self.subset_names = {
            "sentiment_train": None,
            "sentiment_test": None,

            # "diacratization_train": None,
            # "diacratization_test": None,

            # "mcq_train": None,
            # "mcq_test": None,

            "pos_tagging_train": "ar_padt",
            "pos_tagging_test": "ar_padt",

            # "irab_train": "ar_padt",
            # "irab_val": "ar_padt",
            # "irab_test": "ar_padt",

            "paraphrasing_train": None,
            "paraphrasing_test": None,

            "GQA_train": None,
            "GQA_test": None,
        }

        self.prompt_func_map = {

            "dialect_train": self.format_prompt_dialect,
            "dialect_test": self.format_prompt_dialect,

            "irab_train": self.format_prompt_irab,
            "irab_val": self.format_prompt_irab,
            "irab_test": self.format_prompt_irab,

            "translation_train": self.format_prompt_translation,
            "translation_test": self.format_prompt_translation,

            "transliteration_train": self.format_prompt_transliteration,
            "transliteration_test": self.format_prompt_transliteration,
        }

        # =============================================
        self.task_instructions = {
            "summarization": "Can you summarize the following text in one sentence? Give the answer in arabic.",
            "paraphrasing": "Paraphrase the following text while keeping the meaning intact. Give the answer in arabic.",
            "offensive": "Does this text contain offensive language? Type '1' for Offensive and '0' for Not Offensive.",
            "GQA":"What is the answer for the following question?",
            

            "dialect": "",


            "grammar": "Correct the grammatical errors in this sentence",
            # "grammar": "Does this sentence have any grammatical errors? If yes, provide the correction. Otherwise, re-write the sentence",
            # "grammar": "You are a professional proofreader. Read the following sentence and correct any grammatical mistakes",
        }

        self.task_instructions_ar = {
            "sentiment": "صنف مشاعر هذه الجملة كـ 0 إذا كانت سلبية و 1 إذا كانت إيجابية. قم بالاجابة باللغة العربية ",
            "translation": "ترجم الجملة الإنجليزية التالية إلى اللغة العربية",
            "transliteration": "أنت خبير في تحويل النصوص المكتوبة بالأحرف اللاتينية وفقًا لأسلوب العربيزي. حوّل النص التالي إلى الحروف العربية. قم بالاجابة باللغة العربية ",
            #"dialect": "هل كُتب هذا النص باللغة العربية الفصحى أم باللهجة العامية؟ اكتب '0' إذا كان بالفصحى و'1' إذا كان بالعامية.",
            "stance": "حدد الموقف بين الجملتين المعطيتين. اختر أحد التصنيفات التالية: (0) اختلاف، (1) اتفاق، (2) غير واضح/غير مرتبط.",
            "claim": "هل هذا الادعاء زائف؟ اكتب '1' إذا كان زائفًا و'0' إذا لم يكن كذلك.",
            "wsd": "هل يتطابق المعنى المعطى مع معنى الكلمة في هذه الجملة؟ اكتب '1' إذا كان متطابقًا و'0' إذا لم يكن كذلك.",
            "sqs": "هل تمت إعادة صياغة إحدى الجملتين لتكون مكافئة للأخرى؟ أجب بـ '1' إذا كانتا معادتي الصياغة و'0' إذا لم تكونا كذلك.",
            "hate": "صنف هذا النص كـ 0 إذا لم يكن يحتوي على خطاب كراهية و 1 إذا كان يحتوي على خطاب كراهية",
            "pos_tagging": "ما هو النوع الصرفي الصحيح لكل كلمة في هذه الجملة؟ حدد الوسم المناسب لكل كلمة من بين الخيارات التالية: ['NOUN', 'PUNCT', 'ADP', 'NUM', 'SYM', 'SCONJ', 'ADJ', 'PART', 'DET', 'CCONJ', 'PROPN', 'PRON', 'X', 'ADV', 'INTJ', 'VERB', 'AUX'].",
            "irab": "ما هي الوظيفة النحوية (الإعراب) الصحيحة لكل كلمة في هذه الجملة؟ حدد الوسم المناسب لكل كلمة من بين الخيارات التالية: ['acl', 'acl:relcl', 'advcl', 'advmod', 'advmod:emph', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'cc', 'ccomp', 'conj', 'cop', 'csubj', 'csubj:pass', 'dep', 'det', 'discourse', 'dislocated', 'fixed', 'flat:foreign', 'iobj', 'mark', 'nmod', 'nsubj', 'nsubj:pass', 'nummod', 'obj', 'obl', 'obl:arg', 'orphan', 'parataxis', 'punct', 'root', 'xcomp'].",
            "sarcasm": "صنف هذا النص كـ 0 إذا لم يكن ساخراً و 1 إذا كان ساخراً",

            "dialect": "",
            
            "grammar": "صحح الأخطاء النحوية في هذه الجملة",
            # "grammar": "هل تحتوي هذه الجملة على أخطاء نحوية؟ إذا كانت الإجابة نعم، قم بتصحيح الجملة. ان كانت لا تحتوي على اخطاء قم باعادة كتابة الجملة.",
            # "grammar": "أنت مدقق لغوي محترف. اقرأ الجملة التالية وصحح أي أخطاء نحوية"
        }
        # =============================================


        self.size = -1

    def get_size(self):
        assert self.size > 0, "Call get_dataset() first !!!"
        return self.size
    
    def format_prompt_irab(self, data):
        irab_tag_classes = ["acl", "acl:relcl", "advcl", "advmod", "advmod:emph", "amod", "appos", "aux", "aux:pass", "case", "cc", "ccomp", "conj", "cop", "csubj", "csubj:pass", "dep", "det", "discourse", "dislocated", "fixed", "flat:foreign", "iobj", "mark", "nmod", "nsubj", "nsubj:pass", "nummod", "obj", "obl", "obl:arg", "orphan", "parataxis", "punct", "root", "xcomp"]
        tokenized_sents = data["tokens"]
        tags = data["deprels"]
        texts = []

        outputs = []
        for i in range(len(tokenized_sents)):
            tokens = tokenized_sents[i]
            irab_tags = tags[i]

            output = self.convert_to_json(tokens, irab_tags)
            # for j in range(len(tokens)):
            #     output += tokens[j]+":"+irab_tags[j]+"\n"

            outputs.append(output)
            tokenized_sents[i] = " ".join(tokenized_sents[i])


        examples = ""
        shot_inputs = [
            "يذكر ان صاحبي المركزين الاول و الثاني فقط يتأهلان الى سيدني .", 
            "و أضاف التقرير أن ه يجرى حاليا التحقيق مع هؤلاء الاشخاص .", 
            "لأن معظم المصانع التي س يتم إنشاء ها داخل المناطق الحرة .", 
            "و كان المستهدف الذي أعلنت ه الحكومة هو 3 ملايين طن .", 
            "و من المقرر ان تستمر هذه الايام حتى العشرين من الشهر الجاري ."
        ]
        shot_outputs = [
            "{\n"
            "  \"tokens\": [\n"
            "    { \"word\": \"يذكر\", \"label\": \"root\" },\n"
            "    { \"word\": \"ان\", \"label\": \"mark\" },\n"
            "    { \"word\": \"صاحبي\", \"label\": \"nsubj\" },\n"
            "    { \"word\": \"المركزين\", \"label\": \"nmod\" },\n"
            "    { \"word\": \"الاول\", \"label\": \"amod\" },\n"
            "    { \"word\": \"و\", \"label\": \"cc\" },\n"
            "    { \"word\": \"الثاني\", \"label\": \"conj\" },\n"
            "    { \"word\": \"فقط\", \"label\": \"advmod\" },\n"
            "    { \"word\": \"يتأهلان\", \"label\": \"parataxis\" },\n"
            "    { \"word\": \"الى\", \"label\": \"case\" },\n"
            "    { \"word\": \"سيدني\", \"label\": \"obl\" },\n"
            "    { \"word\": \".\", \"label\": \"punct\" }\n"
            "  ]\n"
            "}",

            "{\n"
            "  \"tokens\": [\n"
            "    { \"word\": \"و\", \"label\": \"root\" },\n"
            "    { \"word\": \"أضاف\", \"label\": \"parataxis\" },\n"
            "    { \"word\": \"التقرير\", \"label\": \"nsubj\" },\n"
            "    { \"word\": \"أن\", \"label\": \"mark\" },\n"
            "    { \"word\": \"ه\", \"label\": \"fixed\" },\n"
            "    { \"word\": \"يجرى\", \"label\": \"ccomp\" },\n"
            "    { \"word\": \"حاليا\", \"label\": \"obl\" },\n"
            "    { \"word\": \"التحقيق\", \"label\": \"nsubj\" },\n"
            "    { \"word\": \"مع\", \"label\": \"case\" },\n"
            "    { \"word\": \"هؤلاء\", \"label\": \"det\" },\n"
            "    { \"word\": \"الاشخاص\", \"label\": \"obl:arg\" },\n"
            "    { \"word\": \".\", \"label\": \"punct\" }\n"
            "  ]\n"
            "}",

            "{\n"
            "  \"tokens\": [\n"
            "    { \"word\": \"لأن\", \"label\": \"mark\" },\n"
            "    { \"word\": \"معظم\", \"label\": \"root\" },\n"
            "    { \"word\": \"المصانع\", \"label\": \"nmod\" },\n"
            "    { \"word\": \"التي\", \"label\": \"nsubj\" },\n"
            "    { \"word\": \"س\", \"label\": \"aux\" },\n"
            "    { \"word\": \"يتم\", \"label\": \"acl:relcl\" },\n"
            "    { \"word\": \"إنشاء\", \"label\": \"nsubj\" },\n"
            "    { \"word\": \"ها\", \"label\": \"nmod\" },\n"
            "    { \"word\": \"داخل\", \"label\": \"case\" },\n"
            "    { \"word\": \"المناطق\", \"label\": \"obl\" },\n"
            "    { \"word\": \"الحرة\", \"label\": \"amod\" },\n"
            "    { \"word\": \".\", \"label\": \"punct\" }\n"
            "  ]\n"
            "}",

            "{\n"
            "  \"tokens\": [\n"
            "    { \"word\": \"و\", \"label\": \"root\" },\n"
            "    { \"word\": \"كان\", \"label\": \"cop\" },\n"
            "    { \"word\": \"المستهدف\", \"label\": \"nsubj\" },\n"
            "    { \"word\": \"الذي\", \"label\": \"nsubj\" },\n"
            "    { \"word\": \"أعلنت\", \"label\": \"acl:relcl\" },\n"
            "    { \"word\": \"ه\", \"label\": \"obj\" },\n"
            "    { \"word\": \"الحكومة\", \"label\": \"nsubj\" },\n"
            "    { \"word\": \"هو\", \"label\": \"obl\" },\n"
            "    { \"word\": \"3\", \"label\": \"parataxis\" },\n"
            "    { \"word\": \"ملايين\", \"label\": \"nummod\" },\n"
            "    { \"word\": \"طن\", \"label\": \"nmod\" },\n"
            "    { \"word\": \".\", \"label\": \"punct\" }\n"
            "  ]\n"
            "}",

            "{\n"
            "  \"tokens\": [\n"
            "    { \"word\": \"و\", \"label\": \"root\" },\n"
            "    { \"word\": \"من\", \"label\": \"case\" },\n"
            "    { \"word\": \"المقرر\", \"label\": \"parataxis\" },\n"
            "    { \"word\": \"ان\", \"label\": \"mark\" },\n"
            "    { \"word\": \"تستمر\", \"label\": \"csubj\" },\n"
            "    { \"word\": \"هذه\", \"label\": \"det\" },\n"
            "    { \"word\": \"الايام\", \"label\": \"nsubj\" },\n"
            "    { \"word\": \"حتى\", \"label\": \"case\" },\n"
            "    { \"word\": \"العشرين\", \"label\": \"obl\" },\n"
            "    { \"word\": \"من\", \"label\": \"case\" },\n"
            "    { \"word\": \"الشهر\", \"label\": \"nmod\" },\n"
            "    { \"word\": \"الجاري\", \"label\": \"amod\" },\n"
            "    { \"word\": \".\", \"label\": \"punct\" }\n"
            "  ]\n"
            "}",
        ]
        
        if self.shots > 0:
            examples = self.e_head
            for i in range(self.shots):
                examples += self.q_head + shot_inputs[i] + "\n\n" + self.a_head + "<answer>\n" + shot_outputs[i] + "</answer>\n\n"

        for inp, output in zip(tokenized_sents, outputs):
            text = self.prompt_template.format(examples, inp, output if not self.test_mode else "") + self.EOS_TOKEN
            texts.append(text)

        return {"text": texts}


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

    def read_conllu_file(self, filepath):
        """Parse a .conllu file into a structured list of sentences."""
        examples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                tokens, lemmas, upos, deprels = [], [], [], []
                for tok in sent:
                    # Skip multiword tokens (e.g. "3-4") and empty nodes (e.g. "3.1")
                    if not isinstance(tok["id"], int):
                        continue
                    tokens.append(tok.get("form", ""))
                    lemmas.append(tok.get("lemma", ""))
                    upos.append(tok.get("upostag", ""))
                    deprels.append(tok.get("deprel", ""))

                text = sent.metadata.get("text", " ".join(tokens))
                examples.append({
                    "text": text,
                    "tokens": tokens,
                    "lemmas": lemmas,
                    "upos": upos,
                    "deprels": deprels
                })
        return examples
    
    def convert_to_json(self, tokens, irab_tags):
        items = []
        for w, t in zip(tokens, irab_tags):
            items.append(f'{{ "word": "{w}", "label": "{t}" }}')
        json_text = '{\n  "tokens": [\n    ' + ",\n    ".join(items) + "\n  ]\n}"
        return json_text

    def get_dataset(self, task, lang="ar"):
        self.lang = lang
        print(self.lang, "==========================")

        self.q_head =  "## Question:\n" if self.lang == "en" else (":سؤال" + "##" + "\n")
        self.a_head = "## Response:\n" if self.lang == "en" else (":إجابة" + "##" + "\n")
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
            dataset = load_dataset("json", data_files=self.dataset_names[task_split])["train"]
            
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