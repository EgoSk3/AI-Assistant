import speech_recognition as sr
import requests
import json

import webbrowser
import xml.etree.ElementTree as ET


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
        with sr.Microphone(sample_rate=16000) as source:
            print("\n🎙️ Говорите сейчас... (для выхода скажите 'закончить' или 'остановить')")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )
                return audio
            except sr.WaitTimeoutError:
                print("⏰ Время ожидания истекло")
                return None

    def recognize_speech(self, audio):
        if audio is None:
            print("❌ Аудио не получено")
            return None

        try:
            if self.engine == "whisper":
                result = self.recognizer.recognize_whisper(audio, language="russian", model="base")
                print("🧠 Использован Whisper")
            elif self.engine == "google":
                result = self.recognizer.recognize_google(audio, language="ru-RU")
                print("🧠 Использован Google Web Speech")
            else:
                print(f"⚠️ Неизвестный движок: {self.engine}")
                return None

            return result.strip() if isinstance(result, str) else None

        except Exception as e:
            print(f"❌ Ошибка распознавания ({self.engine}): {str(e)[:100]}")
            return None

    def run(self):
        print(f"=== Система распознавания речи ({self.engine}) ===")
        print("🗣️ Говорите четко и разборчиво. Для выхода скажите 'закончить' или 'остановить'")
        while True:
            audio = self.record_audio()
            if not audio:
                continue

            text = self.recognize_speech(audio)
            if text:
                print(f"\n📝 Результат: \033[1;32m{text}\033[0m")
                if text.lower() in ["закончить", "остановить"]:
                    break
                yield text
            else:
                print("❌ Речь не распознана")


# ================================
# 2. OLLAMA NER Анализатор
# ================================

def build_prompt(text):
    return f"""
Ты — система анализа промышленного оборудования. Извлеки данные из текста в формате JSON, приводя все сущности к начальной форме (единственное число, именительный падеж).

Текст: "{text}"

Правила:
1. equipment: тип оборудования (электродвигатель, пресс, насос и т.д.)
2. number: идентификатор оборудования
3. symptom: проблема (3-5 слов в начальной форме)
4. action: требуемое действие (инфинитив)
5. urgency: нормальный/срочный/критичный (срочный при словах "срочно", "авария")

Пример вывода:
{{
  "equipment": "электродвигатель",
  "number": "12",
  "symptom": "коррозия",
  "action": "проверить",
  "urgency": "нормальный"
}}

Выведи ТОЛЬКО JSON, без других слов или комментариев.
""".strip()


def ask_ollama(prompt, model="phi3"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        raw_response = response.json().get("response", "").strip()

        # Если ответ не начинается с {{ → вернем None
        if not raw_response.startswith("{"):
            print("❌ Модель вернула текст вместо JSON")
            return None

        return raw_response
    else:
        print(f"❌ Ошибка Ollama: {response.status_code}")
        return None


def extract_entities(text):
    prompt = build_prompt(text)
    raw_response = ask_ollama(prompt)

    try:
        clean_json = raw_response.split("```json")[1].split("```")[0] if "```json" in raw_response else raw_response
        return json.loads(clean_json)
    except Exception as e:
        print(f"⚠️ Не удалось распарсить JSON: {e}")
        return {"raw_response": raw_response}


# ================================
# 2. Генерация и открытие ссылки
# ================================

def build_winnum_url(entities):
    base_url = "http://127.0.0.1/Winnum/views/pages/app/agw.jsp"

    params = {
        "rpc": "WNApplicationTagHelper.getTagCalculationValue",
        "mode": "yes",
        "appid": "winnum.org.app.WNApplicationInstance:1",
        "from": "now-2h",
        "till": "now"
    }

    if "equipment" in entities and "number" in entities:
        equip_type = entities["equipment"].lower()
        product_map = {
            "станок": "WNProduct:",
            "пресс": "WNPress:",
            "робот": "WNRobot:",
            "двигатель": "WNMotor:",
            "насос": "WNPump:",
            "электродвигатель": "WNMotor:"
        }
        pid = product_map.get(equip_type, "WNProduct:") + str(entities["number"])
        params["pid"] = pid

    if "symptom" in entities:
        symptom_map = {
            "вибрация": "NC_VIBRATION",
            "температура": "NC_TEMPERATURE",
            "давление": "NC_PRESSURE",
            "шум": "NC_NOISE",
            "перегрев": "NC_OVERHEATING",
            "коррозия": "NC_CORROSION"
        }
        tid = symptom_map.get(entities["symptom"].lower(), "NC_DEFAULT")
        params["tid"] = tid

    query = "&".join([f"{k}={v}" for k, v in params.items()])
    full_url = f"{base_url}?{query}"
    return full_url


# ================================
# 3. Main Loop
# ================================

if __name__ == "__main__":
    print("🚀 Запуск помощника для Winnum / IIoT.Istok")
    print("⌨️ Введите запрос (или 'выход'):")

    while True:
        user_text = input("> ").strip()
        if user_text.lower() in ["выход", "закончить", "остановить"]:
            break

        print("\n🔍 Извлечение ключевых сущностей...")
        entities = extract_entities(user_text)

        print("\n📊 Найденные сущности:")
        for key, value in entities.items():
            print(f"- {key.upper()}: {value}")

        print("\n🌐 Формирование ссылки...")
        full_url = build_winnum_url(entities)
        print(f"\n🔗 Ссылка сформирована: {full_url}")

        print("\n🔗 Открытие в браузере...")
        webbrowser.open(full_url)