import speech_recognition as sr
from datetime import datetime
import requests
import json
import io


class SpeechRecognizer:
    def __init__(self, engine="whisper"):

        self.recognizer = sr.Recognizer()
        self.engine = engine
        self.configure_recognizer()

    def configure_recognizer(self):
        """Настройка параметров распознавания"""
        self.recognizer.pause_threshold = 2.0
        self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = True

    def record_audio(self):
        """Запись аудио с микрофона"""
        with sr.Microphone(sample_rate=16000) as source:
            print("\nГоворите сейчас... (для выхода скажите 'стоп')")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )
                self.save_audio_debug(audio)
                return audio
            except sr.WaitTimeoutError:
                print("Время ожидания истекло")
                return None

    def save_audio_debug(self, audio, prefix="debug"):
        """Сохранение аудио для отладки"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.wav"
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())
        print(f"Аудио сохранено как {filename}")

    def recognize_speech(self, audio):
        """Распознавание речи выбранным движком"""
        if audio is None:
            return None

        try:
            if self.engine == "whisper":
                return self._recognize_whisper(audio)
            elif self.engine == "google":
                return self._recognize_google(audio)
            else:
                print(f"Неизвестный движок: {self.engine}")
                return None
        except Exception as e:
            print(f"Ошибка распознавания ({self.engine}): {str(e)[:100]}")
            return None

    def _recognize_whisper(self, audio):
        """Распознавание через Whisper"""
        result = self.recognizer.recognize_whisper(
            audio,
            language="russian",
            model="base"
        )
        print("Использован Whisper")
        return result.strip() if result else None


    class SpeechRecognizer:
        def __init__(self, engine="whisper"):
            self.recognizer = sr.Recognizer()
            self.engine = engine
            self.configure_recognizer()

        def configure_recognizer(self):
            """Настройка параметров распознавания"""
            self.recognizer.pause_threshold = 3.0
            self.recognizer.energy_threshold = 400
            self.recognizer.dynamic_energy_threshold = True

        def record_audio(self):
            """Запись аудио с микрофона"""
            with sr.Microphone(sample_rate=16000) as source:
                print("\nГоворите сейчас... (Для выхода скажите 'закончить' или 'остановить')")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=5,
                        phrase_time_limit=10
                    )
                    self.save_audio_debug(audio)
                    return audio
                except sr.WaitTimeoutError:
                    print("Время ожидания истекло")
                    return None

        def save_audio_debug(self, audio, prefix="debug"):
            """Сохранение аудио для отладки"""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.wav"
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            print(f"Аудио сохранено как {filename}")

        def recognize_speech(self, audio):
            """Распознавание речи выбранным движком"""
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

                # Проверяем, что результат не None и не пустая строка
                if result and isinstance(result, str):
                    return result.strip()
                return None

            except Exception as e:
                print(f"Ошибка распознавания ({self.engine}): {str(e)[:100]}")
                return None

        def _recognize_whisper(self, audio):
            """Распознавание через Whisper"""
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
            """Распознавание через Google Web Speech"""
            try:
                result = self.recognizer.recognize_google(
                    audio,
                    language="ru-RU",
                    key=None
                )
                print("Использован Google Web Speech")
                return result
            except Exception as e:
                print(f"Google ошибка: {str(e)}")
                return None

        def run(self):
            """Основной цикл распознавания"""
            print(f"=== Система распознавания речи ({self.engine}) ===")
            print("Говорите четко и разборчиво. Для выхода скажите 'закончить' или 'остановить'")

            while True:
                audio = self.record_audio()
                if not audio:
                    continue

                text = self.recognize_speech(audio)
                if text:
                    print(f"\nРезультат: \033[1;32m{text}\033[0m")
                    if text.lower().strip() == "закончить" or text.lower().strip() == "остановить":
                        break
                else:
                    print("Речь не распознана")

    if __name__ == "__main__":
        engine = "whisper"

        try:
            recognizer = SpeechRecognizer(engine=engine)
            recognizer.run()
        except KeyboardInterrupt:
            print("\nПрограмма завершена")