import spacy
from typing import Dict, Any, List, Optional
from datetime import datetime
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
from pathlib import Path


class IIoTAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        # Инициализация NLP модели
        if model_path and Path(model_path).exists():
            self.nlp = spacy.load(model_path)
            print("✅ Загружена существующая NER модель")
        else:
            self.nlp = spacy.blank("ru")
            print("🆕 Создана новая NER модель")

        # Словари для анализа неисправностей
        self.equipment_types = {
            "станок": "Станок", "пресс": "Пресс", "робот": "Робот",
            "конвейер": "Конвейер", "компрессор": "Компрессор"
        }

        self.components = {
            "шпиндель": "Шпиндель", "подшипник": "Подшипник",
            "гидравлика": "Гидравлическая система", "электродвигатель": "Электродвигатель",
            "ремень": "Ремень", "панель": "Панель управления"
        }

        self.symptoms = {
            "шум": "Нехарактерный шум", "вибрация": "Вибрация",
            "перегрев": "Перегрев", "течь": "Течь жидкости",
            "ошибка": "Код ошибки", "заклинивание": "Заклинивание",
            "ремонт": "Требуется ремонт", "остановка": "Остановка работы"
        }

        # Конфигурация NER
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")

    def train_ner_model(self, train_data: List[tuple], output_dir: str = "iiot_ner_model", n_iter: int = 30):
        """Обучение NER модели"""
        # Добавление меток сущностей
        for _, annotations in train_data:
            for ent in annotations.get("entities", []):
                self.ner.add_label(ent[2])

        # Подготовка примеров
        examples = []
        for text, annots in train_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annots)
            examples.append(example)

        # Обучение
        print("🔄 Начало обучения NER модели...")
        optimizer = self.nlp.begin_training()

        for itn in range(n_iter):
            random.shuffle(examples)
            losses = {}
            batches = minibatch(examples, size=compounding(2.0, 16.0, 1.1))
            for batch in batches:
                self.nlp.update(batch, drop=0.3, losses=losses, sgd=optimizer)
            print(f"⏳ Итерация {itn + 1}, Потери: {losses.get('ner', 0):.3f}")

        # Сохранение модели
        self.nlp.to_disk(output_dir)
        print(f"💾 Модель сохранена в '{output_dir}'")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Анализ текста с извлечением сущностей и неисправностей"""
        doc = self.nlp(text)

        # Результаты NER
        ner_result = {
            "equipment": [],
            "equipment_id": [],
            "dates": [],
            "error_codes": [],
            "times": []
        }

        for ent in doc.ents:
            if ent.label_ == "EQUIPMENT":
                ner_result["equipment"].append(ent.text)
                # Попытка извлечь ID оборудования
                id_part = ''.join(c for c in ent.text if c.isdigit())
                if id_part:
                    ner_result["equipment_id"].append(id_part)
            elif ent.label_ == "DATE":
                ner_result["dates"].append(ent.text)
            elif ent.label_ == "ERROR_CODE":
                ner_result["error_codes"].append(ent.text)
            elif ent.label_ == "TIME":
                ner_result["times"].append(ent.text)

        # Анализ неисправностей
        failure_result = {
            "equipment_type": self._get_equipment_type(doc),
            "components": self._get_components(doc),
            "symptoms": self._get_symptoms(doc),
            "urgency": self._detect_urgency(doc),
            "timestamp": self._get_timestamp(doc)
        }

        # Объединение результатов
        return {
            "ner": ner_result,
            "failure_analysis": failure_result,
            "raw_text": text
        }

    def _get_equipment_type(self, doc) -> str:
        """Определение типа оборудования"""
        for token in doc:
            if token.lemma_ in self.equipment_types:
                return self.equipment_types[token.lemma_]
        return "Неизвестное оборудование"

    def _get_components(self, doc) -> List[str]:
        """Поиск неисправных компонентов"""
        found = []
        for token in doc:
            if token.lemma_ in self.components:
                found.append(self.components[token.lemma_])
        return found or ["Не указан"]

    def _get_symptoms(self, doc) -> List[str]:
        """Выявление симптомов"""
        symptoms = []
        text_lower = doc.text.lower()

        for token in doc:
            if token.lemma_ in self.symptoms:
                symptoms.append(self.symptoms[token.lemma_])

        # Дополнительные правила
        if "не включается" in text_lower or "не запускается" in text_lower:
            symptoms.append("Не запускается")
        if "срочный ремонт" in text_lower:
            symptoms.append("Требуется срочный ремонт")

        return symptoms or ["Симптомы не описаны"]

    def _get_timestamp(self, doc) -> str:
        """Извлечение даты/времени"""
        for ent in doc.ents:
            if ent.label_ == "DATE":
                return ent.text
        return datetime.now().strftime("%d.%m.%Y %H:%M")

    def _detect_urgency(self, doc) -> str:
        """Определение срочности"""
        text_lower = doc.text.lower()
        urgency_words = {
            "высокая": ["срочно", "авария", "остановился", "критичн", "срочный"],
            "низкая": ["незначительн", "плановый", "не срочно"]
        }

        for level, words in urgency_words.items():
            if any(word in text_lower for word in words):
                return level.capitalize()
        return "Средняя"

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
    ("Графики нагрузки станка 3 за последние 2 недели", {
        "entities": [(16, 23, "EQUIPMENT"), (27, 44, "DATE")]
    }),
    ("Ошибка E15 на станке 5 в 10:30", {
        "entities": [(7, 10, "ERROR_CODE"), (15, 22, "EQUIPMENT"), (26, 31, "TIME")]
    }),
]


def main():
    print("=" * 60)
    print("🚀 Комбинированная система анализа для IIoT.Istok")
    print("=" * 60)

    # Инициализация анализатора
    analyzer = IIoTAnalyzer()

    # Обучение модели (если нужно)
    if not Path("iiot_ner_model").exists():
        print("\n🛠 Обучение NER модели на тестовых данных...")
        analyzer.train_ner_model(TRAIN_DATA)
        analyzer = IIoTAnalyzer("iiot_ner_model")  # Перезагружаем обученную модель

    # Интерактивный режим
    while True:
        print("\nВведите запрос (или 'выход' для завершения):")
        user_input = input("> ").strip()

        if user_input.lower() in ['выход', 'exit', 'quit']:
            break

        if user_input:
            # Анализ текста
            analysis = analyzer.analyze_text(user_input)
            analyzer.pretty_print_analysis(analysis)

            # Здесь можно добавить логику для работы с API IIoT.Istok
            # Например, получение данных по equipment_id и дате


if __name__ == "__main__":
    main()