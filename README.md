# Модуль А — GUI управления роботом MCX

## Функционал
- Кнопки: Вкл/Выкл/Пауза/Экстренное торможение
- Джойстик MoveJ и MoveL (6 осей)
- Управление вакуумным схватом
- Отображение текущей позы и температуры
- Светофор состояния (синхрон с реальным)
- Полное логирование с сохранением в файл
- Заготовка под видеопоток и ИИ-детекцию

## Запуск
```bash
pip install -r requirements.txt
python main.py


Откройте файл:

text

C:\WPy64-31700\python\Lib\site-packages\mujoco\__init__.py
Найдите строку 239 (ту самую, где ошибка):

Python

_load_all_bundled_plugins()
Замените её на:

try:
    _load_all_bundled_plugins()
except OSError:
    import warnings
    warnings.warn(
        "MuJoCo: не удалось загрузить некоторые плагины. "
        "Базовая функциональность доступна.",
        RuntimeWarning
    )
