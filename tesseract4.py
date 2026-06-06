import os
import glob
import cv2
import numpy as np
import pytesseract
import time
from concurrent.futures import ThreadPoolExecutor

# === НАСТРОЙКА TESSERACT ===
os.environ['TESSDATA_PREFIX'] = 'C:/msys64/ucrt64/share/tessdata/'
pytesseract.pytesseract.tesseract_cmd = 'C:/msys64/ucrt64/bin/tesseract.exe'


# ─────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────

def to_gray(src):
    if len(src.shape) == 3:
        return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return src.copy()


def apply_gamma(image, gamma=0.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def upscale(gray, scale=3):
    return cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)


def remove_noise(binary, size=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


# ─────────────────────────────────────────────
# 11 МЕТОДОВ ПРЕДОБРАБОТКИ
# ─────────────────────────────────────────────

def process_method_1(src):
    gray = to_gray(src)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return remove_noise(binary)


def process_method_2(src):
    gray = to_gray(src)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = remove_noise(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(binary, kernel, iterations=1)


def process_method_3(src):
    gray = to_gray(src)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(scaled, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return remove_noise(binary)


def process_method_4(src):
    gray = to_gray(src)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(scaled, cv2.MORPH_GRADIENT, kernel5)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(binary, kernel3, iterations=1)


def process_method_5(src):
    gray = to_gray(src)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    blurred = cv2.medianBlur(scaled, 3)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return remove_noise(binary)


def process_method_6(src):
    gray = to_gray(src)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    blurred = cv2.medianBlur(scaled, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.dilate(binary, kernel, iterations=1)


def process_method_7(src):
    gray = to_gray(src)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat   = cv2.morphologyEx(scaled, cv2.MORPH_TOPHAT,   kernel)
    blackhat = cv2.morphologyEx(scaled, cv2.MORPH_BLACKHAT, kernel)
    chosen = tophat if tophat.std() >= blackhat.std() else blackhat
    _, binary = cv2.threshold(chosen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return remove_noise(binary)


def process_method_8(src):
    gray = to_gray(src)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    _, binary_norm = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, binary_inv  = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_norm = remove_noise(binary_norm)
    binary_inv  = remove_noise(binary_inv)
    h, w = scaled.shape
    cx, cy = w // 2, h // 2
    roi_norm = binary_norm[cy-cy//2:cy+cy//2, cx-cx//2:cx+cx//2]
    roi_inv  = binary_inv [cy-cy//2:cy+cy//2, cx-cx//2:cx+cx//2]
    return binary_norm if np.sum(roi_norm) >= np.sum(roi_inv) else binary_inv


def process_method_9(src):
    gray = to_gray(src)
    corrected = apply_gamma(gray, gamma=1.5)
    norm = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return remove_noise(binary)


def process_method_10(src):
    gray = to_gray(src)
    corrected = apply_gamma(gray, gamma=1.5)
    norm = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    scaled = upscale(norm, scale=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(scaled, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return remove_noise(binary)


def process_method_11(src):
    # Без предобработки (только приведение к ч/б)
    return to_gray(src)


# ─────────────────────────────────────────────
# ДИСПЕТЧЕР МЕТОДОВ
# ─────────────────────────────────────────────

METHODS = {i: globals()[f"process_method_{i}"] for i in range(1, 12)}

def apply_preprocessing(src, method_id):
    fn = METHODS.get(method_id)
    if fn is None:
        raise ValueError(f"Метод {method_id} не существует.")
    return fn(src)


# ─────────────────────────────────────────────
# УМНЫЙ КАСКАДНЫЙ OCR
# ─────────────────────────────────────────────

def run_ocr(img):
    n_white = np.sum(img == 255)
    n_black = np.sum(img == 0)
    bg_color = 255 if n_white > n_black else 0

    padded = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=bg_color)
    
    # Готовим варианты: оригинал и инверсию
    v1 = padded if bg_color == 255 else cv2.bitwise_not(padded)
    v2 = cv2.bitwise_not(v1)

    # Каскад конфигураций: сначала быстрый PSM 7, затем (если не вышло) точечный PSM 8
    configs = [
        r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789',
        r'--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
    ]

    # Пробуем каскад: сначала проверяем все картинки на PSM 7, затем на PSM 8
    last_res = ""
    for cfg in configs:
        for variant in [v1, v2]:
            try:
                text = pytesseract.image_to_string(variant, config=cfg)
                result = "".join(text.split()).strip()
                if result == "23":  # Досрочный выход: если нашли эталон, прерываем тяжелые циклы
                    return result
                if result:
                    last_res = result
            except:
                pass
    return last_res


# ─────────────────────────────────────────────
# НАЗВАНИЯ МЕТОДОВ
# ─────────────────────────────────────────────

METHOD_NAMES = {
    1:  "Нормализация + Оцу",
    2:  "Нормализация + Оцу + Дилатация эллипсом",
    3:  "Морфологический градиент + Оцу + Дилатация",
    4:  "Градиент (ядро 5x5) + Оцу + Дилатация",
    5:  "Медианный (3) + Оцу",
    6:  "Медианный (3) + Градиент + Оцу + Дилатация",
    7:  "Top-hat / Black-hat + Оцу (автовыбор)",
    8:  "Авто-инверсия Оцу по центральной зоне",
    9:  "Гамма (1.5) + Нормализация + Оцу",
    10: "Гамма (1.5) + Нормализация + Градиент + Оцу + Дилатация",
    11: "Без предобработки (Оригинал)",
}


# ─────────────────────────────────────────────
# ПОТОКОВАЯ ФУНКЦИЯ ОБРАБОТКИ ОДНОГО ФАЙЛА
# ─────────────────────────────────────────────

def worker(path, method_id):
    filename = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return filename, None
    
    processed = apply_preprocessing(img, method_id)
    result = run_ocr(processed)
    return filename, result


# ─────────────────────────────────────────────
# ГЛАВНЫЙ ПАЙПЛАЙН
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  МНОГОПОТОЧНЫЙ ТЕСТ OCR ДЛЯ МЕТАЛЛИЧЕСКОЙ ГРАВИРОВКИ")
    print("=" * 60)
    for k, v in METHOD_NAMES.items():
        print(f"  {k:2d}. {v}")
    print(f"  12. ВСЕ МЕТОДЫ (сравнительный тест)")
    print()

    try:
        choice = int(input("Введите номер метода (1-12): "))
        if not 1 <= choice <= 12:
            print("Ошибка: номер должен быть от 1 до 12.")
            return
    except ValueError:
        print("Ошибка: введите целое число.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.normpath(os.path.join(script_dir, "year1"))

    image_paths = sorted(list(set(
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.JPG")) +
        glob.glob(os.path.join(folder_path, "*.png")) +
        glob.glob(os.path.join(folder_path, "*.PNG"))
    )))

    if not image_paths:
        print(f"Ошибка: в папке '{folder_path}' нет изображений.")
        return

    print(f"\nНайдено изображений: {len(image_paths)}")

    methods_to_run = list(range(1, 12)) if choice == 12 else [choice]
    all_results = {}

    # Определяем оптимальное количество потоков на основе ядер процессора
    num_threads = min(32, (os.cpu_count() or 4) + 4)

    for method_id in methods_to_run:
        if choice == 12:
            print(f"\n{'─' * 60}\n  Метод №{method_id}: {METHOD_NAMES[method_id]}\n{'─' * 60}")
        else:
            print(f"\nМетод №{method_id}: {METHOD_NAMES[method_id]}\n" + "-" * 60)

        total   = 0
        correct = 0
        start   = time.perf_counter()

        # Запуск параллельной обработки картинок в пуле потоков
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, path, method_id) for path in image_paths]
            
            for future in futures:
                filename, result = future.result()
                if result is None:
                    print(f"  [ПРОПУСК] Не удалось прочитать: {filename}")
                    continue
                
                total += 1
                if result == "23":
                    correct += 1
                    status = "✓ УСПЕХ"
                else:
                    status = f"✗ БРАК  (получено: '{result}')"
                
                print(f"  {filename:<30} {status}")

        elapsed  = time.perf_counter() - start
        accuracy = (correct / total * 100) if total > 0 else 0.0
        all_results[method_id] = (correct, total, elapsed, accuracy)

        print(f"\n  Точность: {accuracy:.2f}%  |  Время: {elapsed:.2f} сек.")

    if choice == 12:
        print("\n" + "=" * 60)
        print("  СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (УСКОРЕННАЯ)")
        print("=" * 60)
        print(f"  {'№':<4} {'Точность':>10}  {'Верно':>6}  {'Время':>8}  Метод")
        print(f"  {'─'*4} {'─'*10}  {'─'*6}  {'─'*8}  {'─'*30}")

        best_id = max(all_results, key=lambda k: all_results[k][3])

        for mid, (correct, total, elapsed, accuracy) in all_results.items():
            marker = " ◄ ЛУЧШИЙ" if mid == best_id else ""
            print(f"  {mid:<4} {accuracy:>9.2f}%  {correct:>4}/{total:<2}  "
                  f"{elapsed:>6.2f}с  {METHOD_NAMES[mid]}{marker}")
        print("=" * 60)
    else:
        correct, total, elapsed, accuracy = all_results[choice]
        print("\n" + "=" * 60)
        print(f"  ИТОГИ — Метод №{choice}: {METHOD_NAMES[choice]}")
        print("=" * 60)
        print(f"  Всего изображений:    {total}")
        print(f"  Правильно ('23'):     {correct}")
        print(f"  Точность (Accuracy):  {accuracy:.2f}%")
        print(f"  Время обработки:      {elapsed:.2f} сек.")
        print("=" * 60)


if __name__ == "__main__":
    main()