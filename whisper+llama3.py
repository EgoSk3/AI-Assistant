import speech_recognition as sr
import requests
import json
from datetime import datetime


# ================================
# 1. Speech Recognizer
# ================================

class SpeechRecognizer:
    def __init__(self, engine="whisper"):
        self.recognizer = sr.Recognizer()
        self.engine = engine
        self.configure_recognizer()

    def configure_recognizer(self):
        self.recognizer.pause_threshold = 2.0
        self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = True

    def record_audio(self):
        with sr.Microphone(device_index=4, sample_rate=16000) as source:
            print("\nГоворите сейчас... (для выхода скажите 'закончить' или 'остановить')")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )
                return audio
            except sr.WaitTimeoutError:
                print("Время ожидания истекло")
                return None

    def recognize_speech(self, audio):
        if audio is None:
            print("Аудио не получено")
            return None

        try:
            if self.engine == "whisper":
                result = self._recognize_whisper(audio)
            elif self.engine == "google":
                result = self._recognize_google(audio)
            else:
                print(f"Неизвестный движок: {self.engine}")
                return None

            return result.strip() if isinstance(result, str) else None

        except Exception as e:
            print(f"Ошибка распознавания ({self.engine}): {str(e)[:100]}")
            return None

    def _recognize_whisper(self, audio):
        try:
            result = self.recognizer.recognize_whisper(
                audio,
                language="russian",
                model="base"
            )
            print("Использован Whisper")
            return result
        except Exception as e:
            print(f"Whisper ошибка: {str(e)}")
            return None

    def _recognize_google(self, audio):
        try:
            result = self.recognizer.recognize_google(
                audio,
                language="ru-RU"
            )
            print("Использован Google Web Speech")
            return result
        except Exception as e:
            print(f"Google ошибка: {str(e)}")
            return None

    def run(self):
        print(f"=== Система распознавания речи ({self.engine}) ===")
        print("Говорите четко и разборчиво. Для выхода скажите 'закончить' или 'остановить'")
        while True:
            audio = self.record_audio()
            if not audio:
                continue

            text = self.recognize_speech(audio)
            if text:
                print(f"\nРезультат: \033[1;32m{text}\033[0m")
                if text.lower() in ["закончить", "остановить"]:
                    break
                yield text
            else:
                print("Речь не распознана")


# ================================
# 2. OLLAMA NER Анализатор
# ================================

def build_prompt(text):
    return f"""
Вы — эксперт по промышленным запросам. Ваша задача — извлекать ключевые сущности из текста.
Если встречается что-то новое (неизвестное ранее), добавьте это в ответ с описанием.

Поддерживаемые типы:
- EQUIPMENT — оборудование (станок, робот, пресс)
- NUMBER — номер оборудования
- SYMPTOM — симптом (вибрация, перегрев)
- ERROR_CODE — код ошибки (E45, E78)
- ACTION — действие (диагностика, замена)
- URGENCY — уровень срочности
- DATE — временные метки
- COMPONENT — компонент оборудования
- PART — детали/запчасти

Пример:
"На станке 14 появился треск, срочно диагностика"
→ 
{{
  "equipment": "станок",
  "number": "14",
  "symptom": "треск",
  "action": "диагностика",
  "urgency": "высокая"
}}

Анализируемый текст:
"{text}"

Формат ответа: JSON без дополнительного текста.
""".strip()


def ask_ollama(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        print(f"❌ Ошибка: {response.status_code}")
        return None


def extract_entities(text):
    prompt = build_prompt(text)
    raw_response = ask_ollama(prompt)
    try:
        return json.loads(raw_response)
    except Exception as e:
        print(f"⚠️ Не удалось распарсить JSON: {e}")
        return {"raw_response": raw_response}


# ================================
# 3. Main Loop
# ================================

if __name__ == "__main__":
    recognizer = SpeechRecognizer(engine="whisper")

    print("🚀 Запуск голосового интерфейса для анализа промышленных запросов...")
    print("Скажите ваш запрос:")

    for user_text in recognizer.run():
        print("\n🔍 Извлечение ключевых сущностей...")
        entities = extract_entities(user_text)

        print("\n📊 Найденные сущности:")
        for key, value in entities.items():
            print(f"- {key.upper()}: {value}")

        print("-" * 50)