import spacy
from typing import Dict, Any, List
from datetime import datetime

# Загрузка модели spaCy
nlp = spacy.load("ru_core_news_sm")


class EquipmentFailureAnalyzer:
    def __init__(self):
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

    def analyze(self, text: str) -> Dict[str, Any]:
        """Основной метод анализа текста"""
        doc = nlp(text)

        return {
            "Оборудование": self._get_equipment(doc),
            "Идентификатор": self._get_equipment_id(doc),
            "Неисправные компоненты": self._get_components(doc),
            "Симптомы": self._get_symptoms(doc),
            "Дата возникновения": self._get_timestamp(doc),
            "Срочность": self._detect_urgency(doc),
            "Исходный текст": text
        }

    def _get_equipment(self, doc) -> str:
        """Определение типа оборудования"""
        for token in doc:
            if token.lemma_ in self.equipment_types:
                return self.equipment_types[token.lemma_]
        return "Неизвестное оборудование"

    def _get_equipment_id(self, doc) -> str:
        """Извлечение идентификатора оборудования"""
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PER"] or any(c.isdigit() for c in ent.text):
                return ent.text
        return "Не указан"

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
        for token in doc:
            if token.lemma_ in self.symptoms:
                symptoms.append(self.symptoms[token.lemma_])

        # Дополнительный анализ
        text_lower = doc.text.lower()
        if "не включается" in text_lower or "не запускается" in text_lower:
            symptoms.append("Не запускается")
        if "срочный ремонт" in text_lower:
            symptoms.append("Требуется срочный ремонт")

        return symptoms or ["Симптомы не описаны"]

    def _get_timestamp(self, doc) -> str:
        """Извлечение даты/времени из текста"""
        for ent in doc.ents:
            if ent.label_ == "DATE":
                return ent.text
        return datetime.now().strftime("%d.%m.%Y %H:%M")

    def _detect_urgency(self, doc) -> str:
        """Определение срочности проблемы"""
        text_lower = doc.text.lower()
        if any(word in text_lower for word in ["срочно", "авария", "остановился", "срочный"]):
            return "Высокая"
        elif "незначительный" in text_lower:
            return "Низкая"
        return "Средняя"


# Пример использования
if __name__ == "__main__":
    analyzer = EquipmentFailureAnalyzer()

    print("Система анализа неисправностей оборудования")
    print("Введите описание проблемы:")
    user_input = input("> ")

    result = analyzer.analyze(user_input)

    print("\nРезультат анализа:")
    print("=" * 40)
    for key, value in result.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
    print("=" * 40)