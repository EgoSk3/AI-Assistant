import spacy
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
from pathlib import Path
from collections import defaultdict
import pymorphy3 as pymorphy2
from spacy.tokens import Doc


class IIoTAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        """Инициализация анализатора с морфологическим анализатором и NLP моделью"""
        try:
            self.morph = pymorphy2.MorphAnalyzer()
        except Exception as e:
            print(f"⚠️ Ошибка инициализации морфологического анализатора: {e}")
            self.morph = None

        if model_path and Path(model_path).exists():
            self.nlp = spacy.load(model_path)
            print("✅ Загружена существующая NER модель")
        else:
            self.nlp = spacy.blank("ru")
            print("🆕 Создана новая NER модель")

        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        # Словари с учетом всех склонений и синонимов
        self.equipment_types = {
            "станок": ["станок", "станка", "станку", "станком", "станке", "фрезер", "фрезерный", "токар", "токарный",
                       "шлифовальный", "шлифовщик", "обрабатывающий", "обработка"],
            "пресс": ["пресс", "пресса", "прессу", "прессом", "прессе", "гидропресс", "гидравлический пресс",
                      "кривошипный пресс", "пружинный пресс", "пневматический пресс"],
            "робот": ["робот", "робота", "роботу", "роботом", "роботе", "манипулятор", "сварочный робот",
                      "автоматический робот", "роботизированный", "манипулятор"],
            "конвейер": ["конвейер", "лента", "транспортёр", "транспортер", "поток", "транспортная лента",
                         "ленточный конвейер", "скребковый"],
            "компрессор": ["компрессор", "компрессора", "компрессору", "компрессором", "компрессоре",
                           "воздушный компрессор", "масляный компрессор", "воздушный насос"],
            "линия": ["линия", "линии", "линию", "линией", "лине", "упаковочная линия", "сборочная линия",
                      "автоматическая линия", "комплекс"],
            "печь": ["печь", "печи", "печью", "печкой", "нагревательная установка", "отопительная печь",
                     "кухонная печь"],
            "фрезерный станок": ["фрезер", "фрезерный станок", "фрезерная установка", "фрезерный комплекс"],
            "токарный станок": ["токарь", "токарный станок", "токарка", "обработка по кругу"],
            "шлифовальный станок": ["шлифовальщик", "шлифовальный станок", "шлифовка", "шлифовальная машина"],
            "краскопульт": ["краскопульт", "распылитель", "краскораспылитель", "краскопульту", "краскораспылитель"],
            "балансировочный станок": ["балансировщик", "балансировочный станок", "взвешивающий"],
            "гальваническая ванна": ["гальваническая ванна", "гальваника", "электрохимическая ванна"],
            "сварочный аппарат": ["сварочный аппарат", "сварка", "сварочный комплекс", "сварочный"],
        }

        self.components = {
            "шпиндель": ["шпиндель", "шпинделя", "шпинделю", "шпинделем", "шпинделе", "вал", "торец", "часть вращения"],
            "подшипник": ["подшипник", "подшипника", "подшипнику", "подшипником", "подшипнике", "ролик", "опора"],
            "гидравлика": ["гидравлика", "гидравлики", "гидравлику", "гидравликой", "гидравлике", "насос", "цилиндр",
                           "трубопровод"],
            "электродвигатель": ["электродвигатель", "электродвигателя", "электродвигателю", "электродвигателем",
                                 "электрический мотор", "двигатель"],
            "ремень": ["ремень", "приводной ремень", "клиновой ремень", "зубчатый ремень", "лента", "привод"],
            "датчик": ["датчик", "сенсор", "сигнализатор", "измеритель", "контрольный прибор", "тестер"],
            "трансмиссия": ["трансмиссия", "передача", "редуктор", "механизм передачи"],
            "кабель": ["кабель", "провод", "трос", "жгут", "проводка"],
        }

        self.symptoms = {
            "шум": ["шум", "шума", "шумом", "шуме", "гудение", "скрежет", "скрип", "визг"],
            "вибрация": ["вибрация", "вибрации", "вибрацию", "вибрацией", "тряска", "дребезжание", "колебания"],
            "перегрев": ["перегрев", "перегрева", "перегреву", "перегревом", "жар", "тепловой режим", "перегревание"],
            "ошибка": ["ошибка", "ошибки", "ошибку", "ошибкой", "сбой", "неисправность", "глюк", "проблема"],
            "утечка": ["утечка", "течь", "протечка", "капает", "подтекание", "вытекает", "вылив"],
            "задержка": ["задержка", "задержки", "задержку", "задержкой", "замедление", "зависание"],
            "коррозия": ["коррозия", "ржавчина", "окисление", "загнивание"],
            "засор": ["засор", "загрязнение", "забитость", "засорение"],
        }

        self.actions = {
            "замена": ["замена", "смена", "промывка", "ремонт", "установка", "обслуживание"],
            "ремонт": ["ремонт", "починка", "восстановление", "замена", "обновление"],
            "проверка": ["проверка", "диагностика", "осмотр", "тестирование"],
            "настройка": ["настройка", "регулировка", "калибровка"],
            "заморожен": ["заморожен", "остановлен", "выключен", "заблокирован"],
        }

        self.urgency_keywords = {
            "высокая": ["срочно", "авария", "остановка", "критическая", "критическая ситуация", "аварийная ситуация"],
            "низкая": ["плановая", "плановый", "предупредительный", "профилактика", "регламентное"],
            "средняя": ["нормально", "стандартно", "обычно", "обычная", "обычно"],
        }

        # Настройка NER pipeline
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")

        # Метки для распознавания
        labels = ["EQUIPMENT", "EQUIPMENT_ID", "DATE", "ERROR_CODE", "TIME",
                  "COMPONENT", "SYMPTOM", "ACTION", "URGENCY"]

        for label in labels:
            self.ner.add_label(label)

    def _convert_verb_to_noun(self, word: str) -> Optional[str]:
        """Преобразование глаголов в существительные с обработкой None"""
        if not word:
            return None

        try:
            parsed = self.morph.parse(word)
            if not parsed:
                return None

            # Для глаголов и причастий
            if 'VERB' in parsed[0].tag or 'PRTS' in parsed[0].tag:
                # Пробуем разные формы существительных
                for form in parsed[0].lexeme:
                    if 'NOUN' in form.tag:
                        return form.word

            # Для прилагательных
            if 'ADJF' in parsed[0].tag or 'ADJS' in parsed[0].tag:
                for form in parsed[0].lexeme:
                    if 'NOUN' in form.tag:
                        return form.word

            return word  # Возвращаем исходное слово, если не нашли существительное
        except Exception:
            return word  # В случае ошибки возвращаем исходное слово

    def _normalize(self, word: str) -> str:
        """Приведение слова к нормальной форме с защитой"""
        if not word or not self.morph:
            return word.lower() if word else ""

        try:
            return self.morph.parse(word)[0].normal_form
        except Exception:
            return word.lower()

    def _match_term(self, text: str, term_dict: dict) -> Optional[str]:
        """Поиск термина в словаре с защитой от None"""
        if text is None:
            return None

        text_lower = text.lower()
        normalized = self._normalize(text_lower)

        for term, variants in term_dict.items():
            if normalized == term or any(
                    normalized == self._normalize(variant.lower())
                    for variant in variants
            ):
                return term
        return None

    def _get_equipment_type(self, doc: Doc) -> str:
        """Определение типа оборудования с защитой от None"""
        for token in doc:
            word = self._convert_verb_to_noun(token.text)
            if word:  # Проверяем, что word не None
                term = self._match_term(word, self.equipment_types)
                if term:
                    return term.capitalize()

        # Проверка составных терминов (2 слова)
        for i in range(len(doc) - 1):
            phrase = f"{doc[i].text} {doc[i + 1].text}"
            term = self._match_term(phrase, self.equipment_types)
            if term:
                return term.capitalize()

        return "Неизвестное оборудование"

    def _get_components(self, doc: Doc) -> List[str]:
        """Извлечение компонентов с учетом всех форм"""
        components = set()

        for token in doc:
            word = self._convert_verb_to_noun(token.text)
            component = self._match_term(word, self.components)
            if component:
                components.add(component)
        if not components:
            return ["Не указаны"]
        return list(components)

    def _get_symptoms(self, doc: Doc) -> List[str]:
        """Улучшенное извлечение симптомов с учетом контекста"""
        symptoms = set()

        # 1. Анализ по NER-разметке
        for ent in doc.ents:
            if ent.label_ == "SYMPTOM":
                symptoms.add(ent.text.lower())

        # 2. Анализ по словарю с учетом всех форм слов
        for token in doc:
            # Проверяем само слово
            symptom = self._match_term(token.text, self.symptoms)
            if symptom:
                symptoms.add(symptom)

        # Проверяем лемму слова
            symptom = self._match_term(token.lemma_, self.symptoms)
            if symptom:
                symptoms.add(symptom)

        # 3. Анализ глаголов и причастий
        for token in doc:
            if token.pos_ in ["VERB", "ADJ"]:
                # Пробуем преобразовать в существительное
                noun_form = self._convert_verb_to_noun(token.text)
                if noun_form:
                    symptom = self._match_term(noun_form, self.symptoms)
                    if symptom:
                        symptoms.add(symptom)

        # 4. Анализ составных симптомов (2-3 слова)
        for i in range(len(doc)-1):
            phrase = f"{doc[i].text} {doc[i+1].text}"
            symptom = self._match_term(phrase, self.symptoms)
            if symptom:
                symptoms.add(symptom)

        # 5. Контекстный анализ (если есть sentencizer)
        if "sentencizer" in self.nlp.pipe_names:
            for sent in doc.sents:
                sent_text = sent.text.lower()
                for symptom, variants in self.symptoms.items():
                    if any(variant in sent_text for variant in variants):
                        symptoms.add(symptom)

    # 6. Дополнительные правила
        text_lower = doc.text.lower()
        if any(word in text_lower for word in ["вибрирует", "дрожит", "трясется"]):
            symptoms.add("вибрация")
        if any(word in text_lower for word in ["шумит", "гудит", "скрипит"]):
            symptoms.add("шум")
        if any(word in text_lower for word in ["перегревается", "нагревается"]):
            symptoms.add("перегрев")

        return list(symptoms) if symptoms else ["Симптомы не описаны"]

    def train_ner_model(self, train_data: List[tuple], output_dir: str = "iiot_ner_model", n_iter: int = 150):
        """Обучение с расширенными возможностями"""
        # 1. Добавление всех меток
        labels = set()
        for text, annots in train_data:
            for start, end, label in annots.get("entities", []):
                labels.add(label)

        for label in labels:
            self.ner.add_label(label)

        # 2. Создание примеров
        examples = []
        for text, annots in train_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annots)
            examples.append(example)

        # 3. Обучение с валидацией
        random.shuffle(examples)
        train_examples = examples[:int(0.8 * len(examples))]
        eval_examples = examples[int(0.8 * len(examples)):]

        optimizer = self.nlp.begin_training()

        for itn in range(n_iter):
            losses = {}
            batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.1))

            for batch in batches:
                self.nlp.update(batch, drop=0.4, losses=losses, sgd=optimizer)

            # Оценка точности
            correct = 0
            total = 0
            for eval_ex in eval_examples:
                doc = self.nlp(eval_ex.reference.text)
                gold_ents = {(e.start_char, e.end_char, e.label_) for e in eval_ex.reference.ents}
                pred_ents = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
                correct += len(gold_ents & pred_ents)
                total += len(gold_ents)

            accuracy = correct / total if total > 0 else 0
            print(f"Iter {itn + 1}: Loss={losses['ner']:.3f}, Accuracy={accuracy:.2f}")

        self.nlp.to_disk(output_dir)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Полный анализ текста с комбинированием NER и словарных методов"""
        doc = self.nlp(text)

        # 1. Извлечение сущностей через NER
        ner_result = {
            "equipment": [],
            "equipment_id": [],
            "dates": [],
            "error_codes": [],
            "times": [],
            "components": [],
            "symptoms": [],
            "actions": [],
            "unknown_terms": []  # Для слов, которые не распознаны
        }

        # 2. Извлечение NER-сущностей
        for ent in doc.ents:
            if ent.label_ == "EQUIPMENT":
                ner_result["equipment"].append(ent.text)
                # Автоматическое извлечение ID
                id_part = ''.join(filter(str.isdigit, ent.text))
                if id_part:
                    ner_result["equipment_id"].append(id_part)
            elif ent.label_ == "DATE":
                ner_result["dates"].append(ent.text)
            elif ent.label_ == "ERROR_CODE":
                ner_result["error_codes"].append(ent.text)
            elif ent.label_ == "TIME":
                ner_result["times"].append(ent.text)
            elif ent.label_ == "COMPONENT":
                ner_result["components"].append(ent.text)
            elif ent.label_ == "SYMPTOM":
                ner_result["symptoms"].append(ent.text)
            elif ent.label_ == "ACTION":
                ner_result["actions"].append(ent.text)

        # 3. Поиск оборудования по шаблонам (если NER не нашел)
        if not ner_result["equipment"]:
            for token in doc:
                # Паттерны типа "Слово + цифры" (станок 5, ABC-12)
                if (token.pos_ in ["NOUN", "PROPN"] and
                        any(c.isdigit() for c in token.text)):
                    ner_result["equipment"].append(token.text)
                    id_part = ''.join(filter(str.isdigit, token.text))
                    if id_part:
                        ner_result["equipment_id"].append(id_part)

        # 4. Комбинированный анализ симптомов
        symptoms = set(ner_result["symptoms"])
        for token in doc:
            # Преобразование глаголов в симптомы (вибрирует → вибрация)
            if token.pos_ == "VERB":
                noun_form = self._verb_to_symptom_noun(token.text)
                if noun_form:
                    symptoms.add(noun_form)

        # 5. Определение типа оборудования (комбинированный подход)
        equipment_type = self._determine_equipment_type(doc, ner_result["equipment"])

        # 6. Сборка итогового результата
        failure_result = {
            "equipment_type": equipment_type,
            "components": list(set(ner_result["components"] or self._get_components_from_dict(doc))),
            "symptoms": list(symptoms or self._get_symptoms_from_dict(doc)),
            "urgency": self._detect_urgency(doc),
            "timestamp": self._get_timestamp(doc),
            "unknown_terms": self._find_unknown_terms(doc)  # Для отладки
        }

        return {
            "ner": {k: v for k, v in ner_result.items() if k != "unknown_terms"},
            "failure_analysis": failure_result,
            "raw_text": text,
            "success": any(ner_result.values())
        }

    def _determine_equipment_type(self, doc: Doc, found_equipment: List[str]) -> str:
        """Комбинированное определение типа оборудования"""
        # 1. Попробовать определить из NER-результатов
        if found_equipment:
            for eq in found_equipment:
                # Проверить, содержит ли название известный тип
                for eq_type, variants in self.equipment_types.items():
                    if any(v.lower() in eq.lower() for v in [eq_type] + variants):
                        return eq_type.capitalize()
            # Если не нашли, вернуть первое найденное оборудование
            return found_equipment[0].capitalize()

        # 2. Словарный поиск
        for token in doc:
            eq_type = self._match_term(token.text, self.equipment_types)
            if eq_type:
                return eq_type.capitalize()

        # 3. Поиск по контексту (глаголы поломки + существительное)
        for i, token in enumerate(doc[:-1]):
            if token.lemma_ in ["сломаться", "остановиться", "перегреться"]:
                next_token = doc[i + 1]
                if next_token.pos_ in ["NOUN", "PROPN"]:
                    return next_token.text.capitalize()

        return "Неизвестное оборудование"

    def _verb_to_symptom_noun(self, verb: str) -> Optional[str]:
        """Преобразование глаголов в существительные-симптомы"""
        mapping = {
            "вибрировать": "вибрация",
            "шуметь": "шум",
            "перегреваться": "перегрев",
            "течь": "течь",
            "зависать": "зависание",
            "останавливаться": "остановка"
        }
        parsed = self.morph.parse(verb)[0]
        if parsed.normal_form in mapping:
            return mapping[parsed.normal_form]
        return None

    def _find_unknown_terms(self, doc: Doc) -> List[str]:
        """Поиск терминов, которые не были распознаны"""
        unknown = []
        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN"] and
                    not any(ent.text == token.text for ent in doc.ents)):
                unknown.append(token.text)
        return unknown

    def _get_components_from_dict(self, doc: Doc) -> List[str]:
        """Словарный поиск компонентов"""
        components = set()
        for token in doc:
            component = self._match_term(token.text, self.components)
            if component:
                components.add(component)
        return list(components) if components else ["Не указаны"]

    def _get_symptoms_from_dict(self, doc: Doc) -> List[str]:
        """Словарный поиск симптомов"""
        symptoms = set()
        for token in doc:
            symptom = self._match_term(token.text, self.symptoms)
            if symptom:
                symptoms.add(symptom)
        return list(symptoms) if symptoms else ["Симптомы не описаны"]
    def _detect_urgency(self, doc: Doc) -> str:
        """Определение срочности"""
        text_lower = doc.text.lower()
        urgency_words = {
            "высокая": ["срочно", "авария", "остановка", "критичн"],
            "низкая": ["плановый", "не срочно", "профилактика"]
        }

        for level, words in urgency_words.items():
            if any(word in text_lower for word in words):
                return level.capitalize()
        return "Средняя"

    def _get_timestamp(self, doc: Doc) -> str:
        """Извлечение временной метки"""
        for ent in doc.ents:
            if ent.label_ == "DATE":
                return ent.text
        return datetime.now().strftime("%d.%m.%Y %H:%M")

    def pretty_print_analysis(self, analysis: Dict[str, Any]):
        """Красивый вывод результатов анализа"""
        print(f"\n📋 Анализ запроса: '{analysis['raw_text']}'")
        print("=" * 60)

        print("\n🔍 Извлеченные сущности:")
        for key, values in analysis["ner"].items():
            if values:
                print(f"- {key.upper()}: {', '.join(values)}")

        print("\n⚙️ Анализ неисправностей:")
        failure = analysis["failure_analysis"]
        print(f"- Тип оборудования: {failure['equipment_type']}")
        print(f"- Компоненты: {', '.join(failure['components'])}")
        print(f"- Симптомы: {', '.join(failure['symptoms'])}")
        print(f"- Срочность: {failure['urgency']}")
        print(f"- Временная метка: {failure['timestamp']}")
        print("=" * 60)


# Пример тренировочных данных
TRAIN_DATA = [
    ("Покажи отчет по станку 5 за июнь 2023 года", {
        "entities": [(17, 24, "EQUIPMENT"), (28, 41, "DATE")]
    }),
    ("Почините робот KUKA-5", {"entities": [(10, 17, "EQUIPMENT")]
    }),
    ("Робот Кука-5 сломался", {"entities": [(6, 12, "EQUIPMENT")]
    }),
    ("Графики нагрузки станка 3 за последние 2 недели", {
        "entities": [(16, 23, "EQUIPMENT"), (27, 44, "DATE")]
    }),
    ("Ошибка E15 на станке 5 в 10:30", {
        "entities": [(7, 10, "ERROR_CODE"), (15, 22, "EQUIPMENT"), (26, 31, "TIME")]
    }),
    ("Шпиндель станка 2 вибрирует", {
        "entities": [(0, 8, "COMPONENT"), (9, 16, "EQUIPMENT"), (17, 26, "SYMPTOM")]
    }),
    ("Заменить подшипник на прессе 1", {
        "entities": [(8, 17, "COMPONENT"), (21, 27, "EQUIPMENT"), (0, 8, "ACTION")]
    }),
    ("Требуется ремонт гидравлики робота 3", {
        "entities": [(9, 15, "ACTION"), (16, 26, "COMPONENT"), (27, 33, "EQUIPMENT")]
    }),
    # Добавьте сюда еще 44 примера с вариациями, ошибками, разговорным стилем
    # (Обратите внимание, что для полноты — вам нужно подготовить их самостоятельно или я могу помочь дополнительно)
    ("На станке 12 обнаружена трещина в шпинделе", {
        "entities": [(3, 9, "EQUIPMENT"), (27, 34, "COMPONENT")]
    }),
    ("Гидронасос 4 сбоит, есть утечка масла", {
        "entities": [(0, 10, "EQUIPMENT"), (11, 17, "NUMBER"), (27, 36, "SYMPTOM")]
    }),
    ("Ошибка E78 у электродвигателя 6", {
        "entities": [(8, 11, "ERROR_CODE"), (14, 30, "EQUIPMENT"), (31, 32, "NUMBER")]
    }),
    ("На пресс 2 появилось сильное потрясение", {
        "entities": [(3, 9, "EQUIPMENT"), (10, 11, "NUMBER"), (23, 41, "SYMPTOM")]
    }),
    ("Вибрация и шум на линии 3", {
        "entities": [(0, 8, "SYMPTOM"), (13, 17, "SYMPTOM"), (22, 24, "NUMBER")]
    }),
    ("Проблема с датчиком температуры 7, нужно проверить", {
        "entities": [(11, 17, "COMPONENT"), (18, 19, "NUMBER")]
    }),
    ("Код ошибки E12, станок 9 не работает", {
        "entities": [(14, 17, "ERROR_CODE"), (22, 27, "EQUIPMENT"), (28, 30, "NUMBER")]
    }),
    ("Перегрев гидроцилиндра 4, требуется ремонт", {
        "entities": [(0, 8, "SYMPTOM"), (9, 28, "COMPONENT"), (29, 35, "ACTION")]
    }),
    ("Течь масла у станка 8", {
        "entities": [(0, 4, "SYMPTOM"), (5, 20, "EQUIPMENT"), (21, 22, "NUMBER")]
    }),
    ("Шум и вибрация в роботе 2", {
        "entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (21, 22, "NUMBER")]
    }),
    ("Обнаружена трещина в шпинделе станка 10", {
        "entities": [(16, 29, "COMPONENT"), (30, 36, "EQUIPMENT"), (37, 39, "NUMBER")]
    }),
    ("Ошибка E45 в системе автоматизации", {
        "entities": [(8, 11, "ERROR_CODE"), (12, 37, "EQUIPMENT")]
    }),
    ("На линии 11 вышел из строя приводной ремень", {
        "entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 33, "COMPONENT")]
    }),
    ("Перегрев электродвигателя 5, возможен сбой", {
        "entities": [(0, 8, "SYMPTOM"), (9, 25, "EQUIPMENT"), (26, 31, "NUMBER")]
    }),
    ("Обнаружена утечка в гидравлическом приводе 12", {
        "entities": [(16, 45, "COMPONENT"), (46, 48, "NUMBER")]
    }),
    ("Гудит и трещит гидронасос 3", {
        "entities": [(0, 4, "SYMPTOM"), (5, 15, "SYMPTOM"), (16, 17, "NUMBER")]
    }),
    ("Проблема с датчиком давления, станок 4", {
        "entities": [(11, 23, "COMPONENT"), (24, 30, "EQUIPMENT"), (31, 32, "NUMBER")]
    }),
    ("Код ошибки E56, в системе гидравлики", {
        "entities": [(14, 16, "ERROR_CODE"), (17, 37, "EQUIPMENT")]
    }),
    ("Шпиндель 15 потрескался", {
        "entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 23, "SYMPTOM")]
    }),
    ("На прессе 7 обнаружена трещина", {
        "entities": [(3, 9, "EQUIPMENT"), (10, 11, "NUMBER"), (27, 36, "COMPONENT")]
    }),
    ("Вибрация на линии 8 усиливается", {
        "entities": [(0, 8, "SYMPTOM"), (12, 14, "NUMBER")]
    }),
    ("Ошибка E99 на станке 11", {
        "entities": [(8, 11, "ERROR_CODE"), (15, 21, "EQUIPMENT"), (22, 24, "NUMBER")]
    }),
    ("Обнаружена трещина в гидроцилиндре 13", {
        "entities": [(16, 31, "COMPONENT"), (32, 34, "NUMBER")]
    }),
    ("Проблема с охлаждением у станка 14", {
        "entities": [(11, 20, "COMPONENT"), (21, 27, "EQUIPMENT"), (28, 30, "NUMBER")]
    }),
    ("Датчик температуры 16 вышел из строя", {
        "entities": [(0, 16, "COMPONENT"), (17, 19, "NUMBER")]
    }),
    ("Код ошибки E78, требуется диагностика", {
        "entities": [(8, 11, "ERROR_CODE"), (12, 41, "EQUIPMENT")]
    }),
    ("Гудит и вибрирует гидроцилиндр 17", {
        "entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 19, "NUMBER")]
    }),
    ("Перегрев двигателя 18, возможен сбой", {
        "entities": [(0, 8, "SYMPTOM"), (9, 18, "COMPONENT"), (19, 21, "NUMBER")]
    }),
    ("На станке 19 обнаружена утечка масла", {
        "entities": [(3, 9, "EQUIPMENT"), (10, 27, "SYMPTOM")]
    }),
    ("Обнаружена трещина в шпинделе 20", {
        "entities": [(16, 29, "COMPONENT"), (30, 32, "NUMBER")]
    }),
    ("Ошибка E12 на станке 21", {
        "entities": [(8, 11, "ERROR_CODE"), (15, 21, "EQUIPMENT"), (22, 24, "NUMBER")]
    }),
    ("Шум и вибрация в роботе 22", {
        "entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (21, 23, "NUMBER")]
    }),
    ("Обнаружена утечка масла у пресса 23", {
        "entities": [(16, 29, "SYMPTOM"), (30, 36, "EQUIPMENT"), (37, 39, "NUMBER")]
    }),
    ("Проблема с датчиком давления 24", {
        "entities": [(11, 23, "COMPONENT"), (24, 26, "NUMBER")]
    }),
    ("Код ошибки E56, в системе гидравлики 25", {
        "entities": [(14, 16, "ERROR_CODE"), (17, 41, "EQUIPMENT")]
    }),
    ("Шпиндель 26 потрескался", {
        "entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 23, "SYMPTOM")]
    }),
    ("На прессе 27 обнаружена трещина", {
        "entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "COMPONENT")]
    }),
    ("Вибрация и шум на линии 28 усиливаются", {
        "entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (25, 27, "NUMBER")]
    }),
    ("Ошибка E99 на станке 29", {
        "entities": [(8, 11, "ERROR_CODE"), (15, 21, "EQUIPMENT"), (22, 24, "NUMBER")]
    }),
    ("Обнаружена трещина в гидроцилиндре 30", {
        "entities": [(16, 31, "COMPONENT"), (32, 34, "NUMBER")]
    }),
    ("Проблема с охлаждением у станка 31", {
        "entities": [(11, 20, "COMPONENT"), (21, 27, "EQUIPMENT"), (28, 30, "NUMBER")]
    }),
    ("Датчик температуры 32 вышел из строя", {
        "entities": [(0, 16, "COMPONENT"), (17, 19, "NUMBER")]
    }),
    ("Код ошибки E78, требуется диагностика", {
        "entities": [(8, 11, "ERROR_CODE"), (12, 41, "EQUIPMENT")]
    }),
    ("Гудит и вибрирует гидроцилиндр 33", {
        "entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 19, "NUMBER")]
    }),
    ("Перегрев двигателя 34, возможен сбой", {
        "entities": [(0, 8, "SYMPTOM"), (9, 18, "COMPONENT"), (19, 21, "NUMBER")]
    }),
("Ремень привода на станке 4 изношен и требует замены.", {
        "entities": [(0, 5, "COMPONENT"), (26, 33, "EQUIPMENT")]
    }),
    ("На гидронасосе 3 обнаружена трещина, срочно ремонтировать.", {
        "entities": [(3, 13, "EQUIPMENT"), (24, 33, "COMPONENT")]
    }),
    ("Ошибка E12 у станка 7, необходимо проверить систему.", {
        "entities": [(8, 11, "ERROR_CODE"), (15, 22, "EQUIPMENT")]
    }),
    ("Вибрация и гул в приводе 5, ситуация критическая.", {
        "entities": [(0, 8, "SYMPTOM"), (13, 14, "NUMBER")]
    }),
    ("Обнаружена утечка масла у станка 8, надо срочно устранять.", {
        "entities": [(16, 29, "SYMPTOM"), (30, 36, "EQUIPMENT")]
    }),
    ("Течь охлаждающей жидкости в гидравлическом приводе 12.", {
        "entities": [(0, 24, "SYMPOM"), (25, 45, "COMPONENT")]
    }),
    ("Проблема с датчиком давления на прессе 2.", {
        "entities": [(11, 23, "COMPONENT"), (27, 33, "EQUIPMENT")]
    }),
    ("Код ошибки E45 появился на станке 9, нужно диагностировать.", {
        "entities": [(8, 11, "ERROR_CODE"), (27, 33, "EQUIPMENT")]
    }),
    ("Плохой контакт в электродвигателе 6, проверка необходима.", {
        "entities": [(13, 34, "COMPONENT"), (35, 36, "NUMBER")]
    }),
    ("Гудит и трещит гидроцилиндр 4, требуется ремонт.", {
        "entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 29, "COMPONENT")]
    }),
    ("На линии 10 сломался приводной ремень.", {
        "entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (20, 25, "COMPONENT")]
    }),
    ("Обнаружена трещина в шпинделе станка 11.", {
        "entities": [(16, 29, "COMPONENT"), (30, 36, "EQUIPMENT")]
    }),
    ("Ошибка E78 у электродвигателя 5, требует внимания.", {
        "entities": [(8, 11, "ERROR_CODE"), (14, 30, "EQUIPMENT"), (31, 32, "NUMBER")]
    }),
    ("Перегрев гидронасоса 2, возможно повреждение.", {
        "entities": [(0, 8, "SYMPTOM"), (9, 19, "EQUIPMENT")]
    }),
    ("Шум и вибрация в роботе 4, провести диагностику.", {
        "entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (21, 23, "NUMBER")]
    }),
    ("Обнаружена утечка масла у пресса 15.", {
        "entities": [(16, 29, "SYMPTOM"), (30, 36, "EQUIPMENT"), (37, 39, "NUMBER")]
    }),
    ("Код ошибки E56, в системе гидравлики 16.", {
        "entities": [(14, 16, "ERROR_CODE"), (17, 37, "EQUIPMENT")]
    }),
    ("Шпиндель 17 потрескался, требуется ремонт.", {
        "entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (26, 35, "ACTION")]
    }),
    ("На прессе 18 обнаружена трещина.", {
        "entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (23, 32, "COMPONENT")]
    }),
    ("Вибрация и шум на линии 19 усиливаются.", {
        "entities": [(0, 8, "SYMPTOM"), (13, 17, "SYMPTOM"), (24, 26, "NUMBER")]
    }),
    ("Ошибка E99 на станке 20, нужно проверить систему.", {
        "entities": [(8, 11, "ERROR_CODE"), (15, 21, "EQUIPMENT")]
    }),
    ("Обнаружена трещина в гидроцилиндре 21.", {
        "entities": [(16, 31, "COMPONENT"), (32, 34, "NUMBER")]
    }),
    ("Проблема с охлаждением у станка 22.", {
        "entities": [(11, 20, "COMPONENT"), (21, 23, "NUMBER")]
    }),
    ("Датчик температуры 23 вышел из строя, срочно ремонтировать.", {
        "entities": [(0, 16, "COMPONENT"), (17, 19, "NUMBER")]
    }),
    ("Код ошибки E78, требуется диагностика системы.", {
        "entities": [(8, 11, "ERROR_CODE"), (12, 41, "EQUIPMENT")]
    }),
    ("Гудит и вибрирует гидроцилиндр 24.", {
        "entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 19, "NUMBER")]
    }),
    ("Перегрев двигателя 25, возможен сбой.", {
        "entities": [(0, 8, "SYMPTOM"), (9, 17, "COMPONENT"), (18, 20, "NUMBER")]
    }),
    ("Обнаружена утечка масла у станка 26.", {
        "entities": [(16, 29, "SYMPTOM"), (30, 36, "EQUIPMENT"), (37, 39, "NUMBER")]
    }),
    ("Шум и вибрация в роботе 27.", {
        "entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (21, 23, "NUMBER")]
    }),
    ("Обнаружена трещина в шпинделе 28.", {
        "entities": [(16, 29, "COMPONENT"), (30, 32, "NUMBER")]
    }),
    ("Ошибка E45, станок 29 не работает.", {
        "entities": [(8, 11, "ERROR_CODE"), (15, 21, "EQUIPMENT"), (22, 24, "NUMBER")]
    }),
    ("Шум и гул в приводе 30, ситуация критическая.", {
        "entities": [(0, 3, "SYMPTOM"), (4, 7, "SYMPTOM"), (8, 9, "NUMBER")]
    }),
    ("Обнаружена утечка масла у пресса 31.", {
        "entities": [(16, 29, "SYMPTOM"), (30, 36, "EQUIPMENT"), (37, 39, "NUMBER")]
    }),
    ("Код ошибки E12, в системе автоматизации 32.", {
        "entities": [(8, 11, "ERROR_CODE"), (12, 44, "EQUIPMENT")]
    }),
    ("На станке 33 обнаружена трещина в шпинделе.", {
        "entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "COMPONENT")]
    }),
    ("Вибрация и шум в роботе 34.", {
        "entities": [(0, 8, "SYMPTOM"), (9, 21, "SYMPTOM"), (22, 24, "NUMBER")]
    }),
    ("Ошибка E78 у электродвигателя 35, требует ремонта.", {
        "entities": [(8, 11, "ERROR_CODE"), (14, 34, "EQUIPMENT"), (35, 37, "NUMBER")]
    }),
    ("Перегрев гидронасоса 36, возможна остановка.", {
        "entities": [(0, 8, "SYMPTOM"), (9, 19, "EQUIPMENT")]
    }),
    ("Шум и вибрация в приводе 37, срочно ремонтировать.", {
        "entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (17, 18, "NUMBER")]
    }),
    ("Обнаружена трещина в гидроцилиндре 38.", {
        "entities": [(16, 31, "COMPONENT"), (32, 34, "NUMBER")]
    }),
    ("Проблема с охлаждением у станка 39.", {
        "entities": [(11, 20, "COMPONENT"), (21, 23, "NUMBER")]
    }),
    ("Датчик температуры 40 вышел из строя, срочно ремонтировать.", {
        "entities": [(0, 16, "COMPONENT"), (17, 19, "NUMBER")]
    }),
    ("Код ошибки E56, в системе гидравлики 41.", {
        "entities": [(14, 16, "ERROR_CODE"), (17, 41, "EQUIPMENT")]
    }),
    ("Шпиндель 42 потрескался, требуется замена.", {
        "entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (26, 34, "ACTION")]
    }),
    ("На прессе 43 обнаружена трещина, срочно ремонтировать.", {
        "entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 45, "ACTION")]
    }),
    ("Вибрация и шум на линии 44 усиливаются.", {
        "entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (24, 26, "NUMBER")]
    }),
    ("Ошибка E99 на станке 45, нужно проверить систему.", {
        "entities": [(8, 11, "ERROR_CODE"), (15, 21, "EQUIPMENT")]
    }),
    ("Обнаружена трещина в гидроцилиндре 46.", {
        "entities": [(16, 31, "COMPONENT"), (32, 34, "NUMBER")]
    }),
    ("Проблема с охлаждением у станка 47.", {
        "entities": [(11, 20, "COMPONENT"), (21, 23, "NUMBER")]
    }),
    ("Датчик температуры 48 вышел из строя, срочно ремонтировать.", {
        "entities": [(0, 16, "COMPONENT"), (17, 19, "NUMBER")]
    }),
    ("Код ошибки E78, требуется диагностика системы 49.", {
        "entities": [(8, 11, "ERROR_CODE"), (12, 47, "EQUIPMENT")]
    }),
    ("Гудит и вибрирует гидроцилиндр 50.", {
        "entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 19, "NUMBER")]
    }),
    # Командные запросы с глаголами
    ("Покажи график вибрации станка 5", {"entities": [(0, 6, "ACTION"), (13, 21, "SYMPTOM"), (25, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Открой мне историю работы пресса 2", {"entities": [(0, 6, "ACTION"), (21, 27, "EQUIPMENT"), (28, 29, "NUMBER")]}),
    ("Срочно покажи данные по перегреву двигателя 3", {"entities": [(6, 14, "ACTION"), (18, 27, "SYMPTOM"), (31, 38, "COMPONENT"), (39, 40, "NUMBER")]}),
    ("Нужно найти все ошибки E45 за вчера", {"entities": [(8, 12, "ACTION"), (16, 20, "ERROR_CODE"), (24, 29, "DATE")]}),
    ("Пришли график температуры шпинделя", {"entities": [(0, 5, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT")]}),
    ("Покажи мне текущее состояние робота 1", {"entities": [(0, 6, "ACTION"), (18, 24, "EQUIPMENT"), (25, 26, "NUMBER")]}),
    ("Какие графики доступны по станку 7?", {"entities": [(4, 9, "EQUIPMENT"), (20, 26, "EQUIPMENT"), (27, 28, "NUMBER")]}),
    ("Получи данные по утечке масла", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "COMPONENT")]}),
    ("Анализ вибрации на прессе 3", {"entities": [(0, 6, "ACTION"), (7, 15, "SYMPTOM"), (19, 25, "EQUIPMENT"), (26, 27, "NUMBER")]}),
    ("Пришлите отчет по всем станкам", {"entities": [(0, 5, "ACTION"), (13, 18, "EQUIPMENT")]}),

    # Многосущностные запросы
    ("Покажи графики вибрации и температуры шпинделя станка 4 за последние 24 часа", {
        "entities": [
            (0, 6, "ACTION"), (13, 21, "SYMPTOM"), (25, 35, "SYMPTOM"),
            (36, 45, "COMPONENT"), (49, 55, "EQUIPMENT"), (56, 57, "NUMBER")
        ]
    }),
    ("Нужен анализ шума, вибрации и утечки масла на роботе 2 за май 2024", {
        "entities": [
            (6, 12, "ACTION"), (13, 17, "SYMPTOM"), (18, 26, "SYMPTOM"),
            (27, 35, "SYMPTOM"), (39, 45, "EQUIPMENT"), (46, 47, "NUMBER")
        ]
    }),
    ("Пришлите график температуры двигателя 1 и давления на прессе 3", {
        "entities": [
            (0, 5, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"),
            (34, 35, "NUMBER"), (39, 47, "SYMPTOM"), (51, 57, "EQUIPMENT"), (58, 59, "NUMBER")
        ]
    }),

    # Составные сущности
    ("Станок ГПУ-12 вибрирует, нужен срочный анализ", {
        "entities": [(0, 5, "EQUIPMENT"), (6, 11, "EQUIPMENT_ID"), (12, 20, "SYMPTOM"), (25, 31, "URGENCY"), (32, 41, "ACTION")]}),
    ("Ошибка E78 на станке 5, покажи детали", {
        "entities": [(0, 5, "ERROR_CODE"), (9, 15, "EQUIPMENT"), (16, 17, "NUMBER"), (18, 24, "ACTION")]}),
    ("Датчик давления станка 6 просачивается, требуется ремонт", {
        "entities": [(0, 10, "COMPONENT"), (11, 17, "EQUIPMENT"), (18, 19, "NUMBER"), (20, 31, "SYMPTOM"), (35, 41, "ACTION")]}),
    ("Шум и вибрация на прессе 2 усиливаются, срочно диагностика", {
        "entities": [(0, 4, "SYMPTOM"), (5, 13, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 25, "NUMBER"), (26, 34, "ACTION"), (35, 40, "URGENCY")]}),

    # С временным контекстом
    ("Покажи данные за последний час", {"entities": [(0, 6, "ACTION"), (12, 22, "DATE")]}),
    ("График за вчера и сегодня по роботу 1", {"entities": [(6, 12, "DATE"), (16, 22, "DATE"), (26, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Какая история работы была у станка 7 за 2024 год?", {"entities": [(18, 24, "EQUIPMENT"), (25, 26, "NUMBER"), (27, 37, "DATE")]}),
    ("Данные с 10:00 до 14:00 по станку 8", {"entities": [(5, 10, "TIME"), (14, 19, "TIME"), (23, 29, "EQUIPMENT"), (30, 31, "NUMBER")]}),

    # Срочность и действия
    ("Срочно! Ошибка E45 на станке 3, требует ремонта", {"entities": [(0, 5, "URGENCY"), (7, 10, "ERROR_CODE"), (14, 20, "EQUIPMENT"), (21, 22, "NUMBER"), (26, 32, "ACTION")]}),
    ("Требуется немедленная диагностика утечки масла", {"entities": [(8, 15, "URGENCY"), (16, 27, "ACTION"), (28, 36, "SYMPTOM")]}),
    ("Выполни срочный анализ вибрации на станке 9", {"entities": [(0, 5, "ACTION"), (6, 11, "URGENCY"), (12, 21, "SYMPTOM"), (25, 31, "EQUIPMENT"), (32, 33, "NUMBER")]}),

    # С составными сущностями
    ("Гидронасос 4 сбоит, есть утечка масла", {"entities": [(0, 10, "EQUIPMENT"), (11, 17, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Код ошибки E78 у электродвигателя 6", {"entities": [(8, 11, "ERROR_CODE"), (14, 30, "EQUIPMENT"), (31, 32, "NUMBER")]}),
    ("Робот Кука-5 сломался", {"entities": [(6, 12, "EQUIPMENT")]}),
    ("Перегрев двигателя 18, возможен сбой", {"entities": [(0, 8, "SYMPTOM"), (9, 18, "COMPONENT"), (19, 21, "NUMBER")]}),
    ("ГПУ-12 сильно вибрирует", {"entities": [(0, 5, "EQUIPMENT_ID"), (12, 21, "SYMPTOM")]}),
    ("Робот 3 вышел из строя", {"entities": [(0, 5, "EQUIPMENT"), (6, 7, "NUMBER")]}),
    ("Ошибка E45 в системе автоматизации", {"entities": [(6, 9, "ERROR_CODE"), (10, 35, "EQUIPMENT")]}),

    # Многосущностные запросы
    ("Нужны данные по вибрации, шуму и перегреву на станках 5 и 6 за 24 часа", {
        "entities": [
            (13, 21, "SYMPTOM"), (23, 27, "SYMPTOM"), (29, 37, "SYMPTOM"),
            (42, 44, "EQUIPMENT"), (48, 50, "EQUIPMENT"), (51, 58, "DATE")
        ]
    }),
    ("Покажи график температуры шпинделя и давления гидросистемы станка 7", {
        "entities": [
            (0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"),
            (37, 45, "SYMPTOM"), (46, 55, "COMPONENT"), (59, 65, "EQUIPMENT"), (66, 67, "NUMBER")
        ]
    }),
    ("Анализ зависимости вибрации и температуры на роботе 2", {
        "entities": [
            (0, 6, "ACTION"), (15, 23, "SYMPTOM"), (27, 37, "SYMPTOM"), (41, 47, "EQUIPMENT"), (48, 49, "NUMBER")
        ]
    }),

    # Сложные запросы
    ("За последний час вибрировал пресс 3", {"entities": [(11, 18, "SYMPTOM"), (25, 31, "EQUIPMENT"), (32, 33, "NUMBER")]}),
    ("Покажи график за вчера и сегодня по станку 10", {"entities": [(0, 6, "ACTION"), (12, 18, "DATE"), (22, 27, "DATE"), (31, 37, "EQUIPMENT"), (38, 40, "NUMBER")]}),
    ("Какие графики доступны по агрегату 12?", {"entities": [(4, 9, "EQUIPMENT"), (13, 20, "EQUIPMENT"), (21, 23, "NUMBER")]}),
    ("Нужен отчет по загрузке оборудования за вчера", {"entities": [(5, 10, "ACTION"), (14, 23, "SYMPTOM"), (24, 33, "EQUIPMENT"), (37, 42, "DATE")]}),

    # Примеры с глаголами
    ("Двигатель 14 нагревается, проверь", {"entities": [(0, 7, "COMPONENT"), (8, 10, "NUMBER"), (11, 20, "SYMPTOM"), (24, 29, "ACTION")]}),
    ("Подшипник шпинделя трескается", {"entities": [(0, 9, "COMPONENT"), (10, 18, "COMPONENT"), (19, 28, "SYMPTOM")]}),
    ("Гидронасос 4 просачивается", {"entities": [(0, 10, "EQUIPMENT"), (11, 12, "NUMBER"), (13, 22, "SYMPTOM")]}),
    ("Электродвигатель 6 перегревается", {"entities": [(0, 14, "COMPONENT"), (15, 16, "NUMBER"), (17, 26, "SYMPTOM")]}),
    ("Пресс 2 вышел из строя", {"entities": [(0, 5, "EQUIPMENT"), (6, 7, "NUMBER"), (8, 17, "SYMPTOM")]}),
    ("Робот 3 сломался", {"entities": [(0, 5, "EQUIPMENT"), (6, 7, "NUMBER"), (8, 16, "SYMPTOM")]}),

    # Добавленные примеры
    ("Какие графики есть по шуму на станке 9?", {"entities": [(4, 9, "EQUIPMENT"), (13, 17, "SYMPTOM"), (21, 27, "EQUIPMENT"), (28, 29, "NUMBER")]}),
    ("Покажи график температуры шпинделя станка 10", {"entities": [(0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"), (37, 43, "EQUIPMENT"), (44, 46, "NUMBER")]}),
    ("Нужно проанализировать работу пресса 11 за 2 дня", {"entities": [(5, 13, "ACTION"), (14, 20, "EQUIPMENT"), (21, 23, "NUMBER"), (24, 33, "DATE")]}),
    ("Ошибка E45, открой диагностику по роботу 12", {"entities": [(6, 9, "ERROR_CODE"), (10, 16, "ACTION"), (20, 26, "EQUIPMENT"), (27, 29, "NUMBER")]}),
    ("Гудит и вибрирует гидронасос 13", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("На станке 14 появился треск, срочно диагностика", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (13, 18, "SYMPTOM"), (19, 27, "ACTION"), (28, 33, "URGENCY")]}),
    ("Покажи историю работы станка 15 за май", {"entities": [(0, 6, "ACTION"), (12, 17, "EQUIPMENT"), (18, 20, "NUMBER"), (21, 26, "DATE")]}),
    ("Ошибка E78, требуется ремонт двигателя 16", {"entities": [(6, 9, "ERROR_CODE"), (10, 16, "ACTION"), (17, 25, "COMPONENT"), (26, 28, "NUMBER")]}),
    ("Шум и гул в приводе 17, требуется проверка", {"entities": [(0, 3, "SYMPTOM"), (4, 7, "SYMPTOM"), (8, 14, "EQUIPMENT"), (15, 17, "NUMBER")]}),
    ("Обнаружена утечка масла у станка 18", {"entities": [(10, 18, "SYMPTOM"), (22, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("Датчик температуры 19 вышел из строя", {"entities": [(0, 16, "COMPONENT"), (17, 19, "NUMBER"), (20, 29, "SYMPTOM")]}),
    ("Код ошибки E99, нужно проверить систему", {"entities": [(8, 11, "ERROR_CODE"), (12, 17, "ACTION"), (21, 27, "EQUIPMENT")]}),
    ("Шпиндель 20 потрескался", {"entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 21, "SYMPTOM")]}),
    ("На прессе 21 обнаружена трещина", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Вибрация усиливается на линии 22", {"entities": [(0, 8, "SYMPTOM"), (24, 29, "EQUIPMENT"), (30, 32, "NUMBER")]}),
    ("Ошибка E45 на станке 23, нужно диагностировать", {"entities": [(6, 9, "ERROR_CODE"), (13, 19, "EQUIPMENT"), (20, 22, "NUMBER"), (23, 33, "ACTION")]}),
    ("Гудит и трещит гидроцилиндр 24", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 29, "EQUIPMENT"), (30, 32, "NUMBER")]}),
    ("На линии 25 сломался приводной ремень", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (20, 25, "COMPONENT")]}),
    ("Обнаружена трещина в шпинделе станка 26", {"entities": [(10, 19, "COMPONENT"), (20, 26, "EQUIPMENT"), (27, 29, "NUMBER")]}),
    ("Ошибка E78 у электродвигателя 27, требует внимания", {"entities": [(6, 9, "ERROR_CODE"), (13, 33, "EQUIPMENT"), (34, 36, "NUMBER")]}),
    ("Перегрев гидронасоса 28, возможно повреждение", {"entities": [(0, 8, "SYMPTOM"), (9, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Шум и вибрация в приводе 29, срочно ремонтировать", {"entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 26, "NUMBER"), (27, 35, "ACTION"), (36, 41, "URGENCY")]}),

    # Еще примеры
    ("Покажи график давления масла в прессе 30", {"entities": [(0, 6, "ACTION"), (13, 21, "SYMPTOM"), (22, 28, "COMPONENT"), (32, 37, "EQUIPMENT"), (38, 40, "NUMBER")]}),
    ("Ошибка E45, требуется диагностика системы ГПУ-12", {"entities": [(6, 9, "ERROR_CODE"), (10, 16, "ACTION"), (20, 23, "EQUIPMENT"), (24, 29, "EQUIPMENT_ID")]}),
    ("Обнаружена утечка масла у станка 31", {"entities": [(10, 18, "SYMPTOM"), (22, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("Код ошибки E78, в системе гидравлики 32", {"entities": [(8, 11, "ERROR_CODE"), (12, 20, "EQUIPMENT"), (21, 34, "COMPONENT"), (35, 37, "NUMBER")]}),
    ("Шпиндель 33 потрескался", {"entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 21, "SYMPTOM")]}),
    ("На прессе 34 обнаружена трещина", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Вибрация и шум на линии 35 усиливаются", {"entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 26, "NUMBER")]}),
    ("Ошибка E99 на станке 36, нужно проверить систему", {"entities": [(6, 9, "ERROR_CODE"), (13, 19, "EQUIPMENT"), (20, 22, "NUMBER"), (23, 33, "ACTION")]}),
    ("Обнаружена трещина в гидроцилиндре 37", {"entities": [(10, 19, "COMPONENT"), (20, 35, "COMPONENT"), (36, 38, "NUMBER")]}),
    ("Проблема с охлаждением у станка 38", {"entities": [(11, 20, "COMPONENT"), (21, 27, "EQUIPMENT"), (28, 30, "NUMBER")]}),
    ("Датчик температуры 39 вышел из строя", {"entities": [(0, 16, "COMPONENT"), (17, 19, "NUMBER"), (20, 29, "SYMPTOM")]}),
    ("Код ошибки E78, требуется диагностика системы 40", {"entities": [(8, 11, "ERROR_CODE"), (12, 20, "ACTION"), (21, 27, "EQUIPMENT"), (31, 33, "NUMBER")]}),
    ("Гудит и вибрирует гидроцилиндр 41", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 29, "EQUIPMENT"), (30, 32, "NUMBER")]}),
    ("Перегрев двигателя 42, возможен сбой", {"entities": [(0, 8, "SYMPTOM"), (9, 17, "COMPONENT"), (18, 20, "NUMBER")]}),
    ("Обнаружена утечка масла у пресса 43", {"entities": [(10, 18, "SYMPTOM"), (22, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("Код ошибки E12, в системе автоматизации 44", {"entities": [(8, 11, "ERROR_CODE"), (12, 44, "EQUIPMENT")]}),
    ("Шпиндель 45 потрескался", {"entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 21, "SYMPTOM")]}),
    ("На станке 46 обнаружена трещина", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Вибрация и шум в роботе 47 усиливаются", {"entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 26, "NUMBER")]}),
    ("Ошибка E78 у станка 48, требует ремонта", {"entities": [(6, 9, "ERROR_CODE"), (13, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Перегрев гидронасоса 49, возможна остановка", {"entities": [(0, 8, "SYMPTOM"), (9, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Шум и вибрация в приводе 50, срочно ремонтировать", {"entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (20, 26, "EQUIPMENT"), (27, 29, "NUMBER")]}),
    ("Обнаружена трещина в гидроцилиндре 51", {"entities": [(10, 19, "COMPONENT"), (20, 35, "COMPONENT"), (36, 38, "NUMBER")]}),
    ("Проблема с охлаждением у станка 52", {"entities": [(11, 20, "COMPONENT"), (21, 27, "EQUIPMENT"), (28, 30, "NUMBER")]}),
    ("Датчик температуры 53 вышел из строя", {"entities": [(0, 14, "COMPONENT"), (15, 17, "NUMBER"), (18, 27, "SYMPTOM")]}),
    ("Код ошибки E56, требуется диагностика системы 54", {"entities": [(8, 11, "ERROR_CODE"), (12, 20, "ACTION"), (21, 27, "EQUIPMENT")]}),
    ("Шпиндель 55 потрескался", {"entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 21, "SYMPTOM")]}),
    ("На прессе 56 обнаружена трещина", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Вибрация и шум в роботе 57 усиливаются", {"entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 26, "NUMBER")]}),
    ("Ошибка E99 на станке 58, нужно проверить систему", {"entities": [(6, 9, "ERROR_CODE"), (13, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Обнаружена утечка масла у станка 59", {"entities": [(10, 18, "SYMPTOM"), (22, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("Код ошибки E12, в системе гидравлики 60", {"entities": [(8, 11, "ERROR_CODE"), (12, 41, "EQUIPMENT")]}),
    ("Шпиндель 61 потрескался", {"entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 21, "SYMPTOM")]}),
    ("На станке 62 обнаружена трещина", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Вибрация и шум в роботе 63 усиливаются", {"entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 26, "NUMBER")]}),
    ("Ошибка E78 у станка 64, требует ремонта", {"entities": [(6, 9, "ERROR_CODE"), (13, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Перегрев гидронасоса 65, возможна остановка", {"entities": [(0, 8, "SYMPTOM"), (9, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Шум и вибрация в приводе 66, срочно ремонтировать", {"entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (20, 26, "EQUIPMENT"), (27, 29, "NUMBER")]}),
    ("Обнаружена трещина в гидроцилиндре 67", {"entities": [(10, 19, "COMPONENT"), (20, 35, "COMPONENT"), (36, 38, "NUMBER")]}),
    ("Проблема с охлаждением у станка 68", {"entities": [(11, 20, "COMPONENT"), (21, 27, "EQUIPMENT"), (28, 30, "NUMBER")]}),
    ("Датчик температуры 69 вышел из строя", {"entities": [(0, 14, "COMPONENT"), (15, 17, "NUMBER"), (18, 27, "SYMPTOM")]}),
    ("Код ошибки E78, требуется диагностика системы 70", {"entities": [(8, 11, "ERROR_CODE"), (12, 20, "ACTION"), (21, 27, "EQUIPMENT")]}),
    ("Шпиндель 71 потрескался", {"entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 21, "SYMPTOM")]}),
    ("На прессе 72 обнаружена трещина", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Вибрация и шум в роботе 73 усиливаются", {"entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 26, "NUMBER")]}),
    ("Ошибка E45 у станка 74, нужно диагностировать", {"entities": [(6, 9, "ERROR_CODE"), (13, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Обнаружена утечка масла у станка 75", {"entities": [(10, 18, "SYMPTOM"), (22, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("Код ошибки E12, в системе гидравлики 76", {"entities": [(8, 11, "ERROR_CODE"), (12, 41, "EQUIPMENT")]}),
    ("Шпиндель 77 потрескался", {"entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 21, "SYMPTOM")]}),
    ("На станке 78 обнаружена трещина", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Вибрация и шум в роботе 79 усиливаются", {"entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 26, "NUMBER")]}),
    ("Ошибка E78 у станка 80, требует ремонта", {"entities": [(6, 9, "ERROR_CODE"), (13, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Перегрев гидронасоса 81, возможна остановка", {"entities": [(0, 8, "SYMPTOM"), (9, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Шум и вибрация в приводе 82, срочно ремонтировать", {"entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (20, 26, "EQUIPMENT"), (27, 29, "NUMBER")]}),
    ("Обнаружена трещина в гидроцилиндре 83", {"entities": [(10, 19, "COMPONENT"), (20, 35, "COMPONENT"), (36, 38, "NUMBER")]}),
    ("Проблема с охлаждением у станка 84", {"entities": [(11, 20, "COMPONENT"), (21, 27, "EQUIPMENT"), (28, 30, "NUMBER")]}),
    ("Датчик температуры 85 вышел из строя", {"entities": [(0, 14, "COMPONENT"), (15, 17, "NUMBER"), (18, 27, "SYMPTOM")]}),
    ("Код ошибки E78, требуется диагностика системы 86", {"entities": [(8, 11, "ERROR_CODE"), (12, 20, "ACTION"), (21, 27, "EQUIPMENT")]}),
    ("Шпиндель 87 потрескался", {"entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 21, "SYMPTOM")]}),
    ("На прессе 88 обнаружена трещина", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Вибрация и шум в роботе 89 усиливаются", {"entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 26, "NUMBER")]}),
    ("Ошибка E99 на станке 90, нужно проверить систему", {"entities": [(6, 9, "ERROR_CODE"), (13, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Обнаружена утечка масла у станка 91", {"entities": [(10, 18, "SYMPTOM"), (22, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("Код ошибки E12, в системе гидравлики 92", {"entities": [(8, 11, "ERROR_CODE"), (12, 41, "EQUIPMENT")]}),
    ("Шпиндель 93 потрескался", {"entities": [(0, 8, "COMPONENT"), (9, 11, "NUMBER"), (12, 21, "SYMPTOM")]}),
    ("На станке 94 обнаружена трещина", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (27, 36, "SYMPTOM")]}),
    ("Вибрация и шум в роботе 95 усиливаются", {"entities": [(0, 8, "SYMPTOM"), (9, 13, "SYMPTOM"), (17, 23, "EQUIPMENT"), (24, 26, "NUMBER")]}),
    ("Ошибка E45 у станка 96, требует диагностики", {"entities": [(6, 9, "ERROR_CODE"), (13, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
    ("Перегрев двигателя 97, возможен сбой", {"entities": [(0, 8, "SYMPTOM"), (9, 17, "COMPONENT"), (18, 20, "NUMBER")]}),
    ("Шум и вибрация в приводе 98, срочно ремонтировать", {"entities": [(0, 3, "SYMPTOM"), (4, 16, "SYMPTOM"), (20, 26, "EQUIPMENT"), (27, 29, "NUMBER")]}),
    ("Обнаружена трещина в гидроцилиндре 99", {"entities": [(10, 19, "COMPONENT"), (20, 35, "COMPONENT"), (36, 38, "NUMBER")]}),
    ("Проблема с датчиком давления 100", {"entities": [(11, 23, "COMPONENT"), (24, 27, "NUMBER")]}),
("Покажи график вибрации станка 5", {"entities": [(0, 6, "ACTION"), (13, 21, "SYMPTOM"), (25, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Открой историю работы пресса 2", {"entities": [(0, 6, "ACTION"), (17, 23, "EQUIPMENT"), (24, 25, "NUMBER")]}),
    ("Срочно покажи данные по перегреву двигателя 3", {"entities": [(6, 14, "ACTION"), (18, 27, "SYMPTOM"), (31, 38, "COMPONENT"), (39, 40, "NUMBER")]}),

    # 📊 Многосущностные запросы
    ("Покажи графики вибрации и температуры шпинделя станка 4 за последние 24 часа", {"entities": [(0, 6, "ACTION"), (13, 21, "SYMPTOM"), (25, 35, "SYMPTOM"), (36, 45, "COMPONENT"), (49, 55, "EQUIPMENT"), (56, 57, "NUMBER")]}),
    ("Нужен анализ шума, вибрации и утечки масла на роботе 2 за май 2024", {"entities": [(5, 13, "ACTION"), (14, 18, "SYMPTOM"), (19, 27, "SYMPTOM"), (28, 36, "SYMPTOM"), (40, 46, "EQUIPMENT"), (47, 48, "NUMBER"), (49, 59, "DATE")]}),

    # ⏰ Временные запросы
    ("Покажи данные за последний час", {"entities": [(0, 6, "ACTION"), (12, 22, "DATE")]}),
    ("График за вчера и сегодня по роботу 1", {"entities": [(6, 12, "DATE"), (16, 22, "DATE"), (26, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Какая история работы была у станка 7 за 2024 год?", {"entities": [(18, 24, "EQUIPMENT"), (25, 26, "NUMBER"), (27, 37, "DATE")]}),

    # ⚠️ Срочность и действия
    ("Срочно! Ошибка E45 на станке 3, требует ремонта", {"entities": [(0, 5, "URGENCY"), (7, 10, "ERROR_CODE"), (14, 20, "EQUIPMENT"), (21, 22, "NUMBER"), (26, 32, "ACTION")]}),
    ("Шум и вибрация усиливаются, срочно диагностика", {"entities": [(0, 4, "SYMPTOM"), (5, 13, "SYMPTOM"), (26, 34, "ACTION"), (35, 40, "URGENCY")]}),

    # 📈 Сложные запросы
    ("Анализ зависимости вибрации и температуры на роботе 2", {"entities": [(0, 6, "ACTION"), (15, 23, "SYMPTOM"), (27, 37, "SYMPTOM"), (41, 47, "EQUIPMENT"), (48, 49, "NUMBER")]}),
    ("Покажи график температуры шпинделя и давления гидросистемы станка 7", {"entities": [(0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"), (37, 45, "SYMPTOM"), (46, 55, "COMPONENT"), (59, 65, "EQUIPMENT"), (66, 67, "NUMBER")]}),

    # 📅 Даты и интервалы
    ("Данные с 10:00 до 14:00 по станку 8", {"entities": [(5, 10, "TIME"), (14, 19, "TIME"), (23, 29, "EQUIPMENT"), (30, 31, "NUMBER")]}),
    ("За последний час вибрировал пресс 3", {"entities": [(11, 18, "SYMPTOM"), (25, 31, "EQUIPMENT"), (32, 33, "NUMBER")]}),

    # 🛠 Оборудование и компоненты
    ("Робот 3 сломался", {"entities": [(0, 5, "EQUIPMENT"), (6, 7, "NUMBER"), (8, 16, "SYMPTOM")]}),
    ("Датчик давления станка 6 просачивается, требуется ремонт", {"entities": [(0, 10, "COMPONENT"), (11, 17, "EQUIPMENT"), (18, 19, "NUMBER"), (20, 31, "SYMPTOM"), (35, 41, "ACTION")]}),

    # 📌 Синонимы глаголов
    ("Найди мне информацию о температуре шпинделя", {"entities": [(0, 5, "ACTION"), (14, 24, "SYMPTOM"), (25, 33, "COMPONENT")]}),
    ("Выведи график вибрации на экран", {"entities": [(0, 6, "ACTION"), (7, 15, "SYMPTOM")]}),
    ("Пришли данные по утечке масла", {"entities": [(0, 5, "ACTION"), (9, 17, "SYMPTOM")]}),

    # 🔄 Разнообразие глаголов
    ("Получи данные по утечке масла", {"entities": [(0, 5, "ACTION"), (9, 17, "SYMPTOM")]}),
    ("Отправь график температуры шпинделя", {"entities": [(0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT")]}),
    ("Загрузи историю работы станка 7", {"entities": [(0, 6, "ACTION"), (12, 17, "EQUIPMENT"), (18, 19, "NUMBER")]}),

    # 📈 Многосущностные запросы с временными метками
    ("График за вчера и сегодня по роботу 1", {"entities": [(6, 12, "DATE"), (16, 22, "DATE"), (26, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Покажи график температуры шпинделя станка 10", {"entities": [(0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"), (37, 43, "EQUIPMENT"), (44, 46, "NUMBER")]}),

    # 📉 Симптомы и компоненты
    ("Гудит и вибрирует гидронасос 13", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("На станке 14 появился треск, срочно диагностика", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (13, 18, "SYMPTOM"), (19, 27, "ACTION"), (28, 33, "URGENCY")]}),

    # 📁 Документы и отчеты
    ("Пришлите отчет по загрузке оборудования за вчера", {"entities": [(0, 5, "ACTION"), (6, 11, "EQUIPMENT"), (15, 24, "SYMPTOM"), (28, 33, "DATE")]}),
    ("Создай таблицу с данными по ошибкам E45", {"entities": [(0, 6, "ACTION"), (11, 16, "EQUIPMENT"), (20, 24, "ERROR_CODE")]}),

    # 📂 Дополнительные примеры
    ("Какие графики есть по шуму на станке 9?", {"entities": [(4, 9, "EQUIPMENT"), (13, 17, "SYMPTOM"), (21, 27, "EQUIPMENT"), (28, 29, "NUMBER")]}),
    ("Ошибка E78 у электродвигателя 27, требует внимания", {"entities": [(6, 9, "ERROR_CODE"), (13, 33, "EQUIPMENT"), (34, 36, "NUMBER")]}),
    ("Перегрев гидронасоса 28, возможно повреждение", {"entities": [(0, 8, "SYMPTOM"), (9, 19, "EQUIPMENT"), (20, 22, "NUMBER")]}),
("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📉 Симптомы и компоненты (новые комбинации)
    ("Гудит и вибрирует гидронасос 13", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("На станке 14 появился треск, срочно диагностика", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (13, 18, "SYMPTOM"), (19, 27, "ACTION"), (28, 33, "URGENCY")]}),

    # 📊 Документы и отчеты (новые сценарии)
    ("Пришлите отчет по загрузке оборудования за вчера", {"entities": [(0, 5, "ACTION"), (6, 11, "EQUIPMENT"), (15, 24, "SYMPTOM"), (28, 33, "DATE")]}),
    ("Создай таблицу с данными по ошибкам E45", {"entities": [(0, 6, "ACTION"), (11, 16, "EQUIPMENT"), (20, 24, "ERROR_CODE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📊 Многосущностные запросы с временными метками
    ("График за вчера и сегодня по роботу 1", {"entities": [(6, 12, "DATE"), (16, 22, "DATE"), (26, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Покажи график температуры шпинделя станка 10", {"entities": [(0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"), (37, 43, "EQUIPMENT"), (44, 46, "NUMBER")]}),

    # 📉 Симптомы и компоненты (новые комбинации)
    ("Гудит и вибрирует гидронасос 13", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("На станке 14 появился треск, срочно диагностика", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (13, 18, "SYMPTOM"), (19, 27, "ACTION"), (28, 33, "URGENCY")]}),

    # 📁 Документы и отчеты (новые сценарии)
    ("Пришлите отчет по загрузке оборудования за вчера", {"entities": [(0, 5, "ACTION"), (6, 11, "EQUIPMENT"), (15, 24, "SYMPTOM"), (28, 33, "DATE")]}),
    ("Создай таблицу с данными по ошибкам E45", {"entities": [(0, 6, "ACTION"), (11, 16, "EQUIPMENT"), (20, 24, "ERROR_CODE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📊 Многосущностные запросы с временными метками
    ("График за вчера и сегодня по роботу 1", {"entities": [(6, 12, "DATE"), (16, 22, "DATE"), (26, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Покажи график температуры шпинделя станка 10", {"entities": [(0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"), (37, 43, "EQUIPMENT"), (44, 46, "NUMBER")]}),

    # 📉 Симптомы и компоненты (новые комбинации)
    ("Гудит и вибрирует гидронасос 13", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("На станке 14 появился треск, срочно диагностика", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (13, 18, "SYMPTOM"), (19, 27, "ACTION"), (28, 33, "URGENCY")]}),

    # 📁 Документы и отчеты (новые сценарии)
    ("Пришлите отчет по загрузке оборудования за вчера", {"entities": [(0, 5, "ACTION"), (6, 11, "EQUIPMENT"), (15, 24, "SYMPTOM"), (28, 33, "DATE")]}),
    ("Создай таблицу с данными по ошибкам E45", {"entities": [(0, 6, "ACTION"), (11, 16, "EQUIPMENT"), (20, 24, "ERROR_CODE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📊 Многосущностные запросы с временными метками
    ("График за вчера и сегодня по роботу 1", {"entities": [(6, 12, "DATE"), (16, 22, "DATE"), (26, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Покажи график температуры шпинделя станка 10", {"entities": [(0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"), (37, 43, "EQUIPMENT"), (44, 46, "NUMBER")]}),

    # 📉 Симптомы и компоненты (новые комбинации)
    ("Гудит и вибрирует гидронасос 13", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("На станке 14 появился треск, срочно диагностика", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (13, 18, "SYMPTOM"), (19, 27, "ACTION"), (28, 33, "URGENCY")]}),

    # 📁 Документы и отчеты (новые сценарии)
    ("Пришлите отчет по загрузке оборудования за вчера", {"entities": [(0, 5, "ACTION"), (6, 11, "EQUIPMENT"), (15, 24, "SYMPTOM"), (28, 33, "DATE")]}),
    ("Создай таблицу с данными по ошибкам E45", {"entities": [(0, 6, "ACTION"), (11, 16, "EQUIPMENT"), (20, 24, "ERROR_CODE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📊 Многосущностные запросы с временными метками
    ("График за вчера и сегодня по роботу 1", {"entities": [(6, 12, "DATE"), (16, 22, "DATE"), (26, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Покажи график температуры шпинделя станка 10", {"entities": [(0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"), (37, 43, "EQUIPMENT"), (44, 46, "NUMBER")]}),

    # 📉 Симптомы и компоненты (новые комбинации)
    ("Гудит и вибрирует гидронасос 13", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("На станке 14 появился треск, срочно диагностика", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (13, 18, "SYMPTOM"), (19, 27, "ACTION"), (28, 33, "URGENCY")]}),

    # 📁 Документы и отчеты (новые сценарии)
    ("Пришлите отчет по загрузке оборудования за вчера", {"entities": [(0, 5, "ACTION"), (6, 11, "EQUIPMENT"), (15, 24, "SYMPTOM"), (28, 33, "DATE")]}),
    ("Создай таблицу с данными по ошибкам E45", {"entities": [(0, 6, "ACTION"), (11, 16, "EQUIPMENT"), (20, 24, "ERROR_CODE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📊 Многосущностные запросы с временными метками
    ("График за вчера и сегодня по роботу 1", {"entities": [(6, 12, "DATE"), (16, 22, "DATE"), (26, 32, "EQUIPMENT"), (33, 34, "NUMBER")]}),
    ("Покажи график температуры шпинделя станка 10", {"entities": [(0, 6, "ACTION"), (13, 23, "SYMPTOM"), (24, 33, "COMPONENT"), (37, 43, "EQUIPMENT"), (44, 46, "NUMBER")]}),

    # 📉 Симптомы и компоненты (новые комбинации)
    ("Гудит и вибрирует гидронасос 13", {"entities": [(0, 4, "SYMPTOM"), (5, 17, "SYMPTOM"), (18, 28, "EQUIPMENT"), (29, 31, "NUMBER")]}),
    ("На станке 14 появился треск, срочно диагностика", {"entities": [(3, 9, "EQUIPMENT"), (10, 12, "NUMBER"), (13, 18, "SYMPTOM"), (19, 27, "ACTION"), (28, 33, "URGENCY")]}),

    # 📁 Документы и отчеты (новые сценарии)
    ("Пришлите отчет по загрузке оборудования за вчера", {"entities": [(0, 5, "ACTION"), (6, 11, "EQUIPMENT"), (15, 24, "SYMPTOM"), (28, 33, "DATE")]}),
    ("Создай таблицу с данными по ошибкам E45", {"entities": [(0, 6, "ACTION"), (11, 16, "EQUIPMENT"), (20, 24, "ERROR_CODE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),

    # 📈 Сложные запросы с API-спецификой
    ("Получи данные по сигналам E45 за последние 15 минут", {"entities": [(5, 9, "ACTION"), (13, 21, "SYMPTOM"), (22, 26, "ERROR_CODE"), (30, 43, "DATE")]}),
    ("Запрос на публикацию исторических данных по станку 7", {"entities": [(0, 6, "ACTION"), (28, 34, "EQUIPMENT"), (35, 36, "NUMBER")]}),
    ("Проверь готовность данных по роботу 2 за 2024 год", {"entities": [(0, 5, "ACTION"), (18, 24, "EQUIPMENT"), (25, 27, "NUMBER"), (28, 38, "DATE")]}),
    ]



def main():
    print("=" * 60)
    print("🚀 Система анализа промышленного оборудования")
    print("=" * 60)

    if not Path("iiot_ner_model").exists():
        print("\n🔎 Обученная модель не найдена, начинаем обучение...")
        analyzer = IIoTAnalyzer()
        analyzer.train_ner_model(TRAIN_DATA)
    else:
        print("\n🔎 Загружаем существующую модель")
        analyzer = IIoTAnalyzer("iiot_ner_model")

    # Тестовые запросы
    test_queries = [
        "Обнаружена коррозия в электродвигателе 12",
        "Покажи график вибрации станка 5"

    ]

    for query in test_queries:
        print("\n" + "=" * 50)
        print(f"Тестовый запрос: {query}")
        analysis = analyzer.analyze_text(query)
        analyzer.pretty_print_analysis(analysis)


if __name__ == "__main__":
    main()
