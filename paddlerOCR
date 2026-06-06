import os
import glob
import cv2
import numpy as np
import Levenshtein
import time
import re
from paddleocr import PaddleOCR


def apply_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def extract_digits(text):
    if not text:
        return ""
    digits = re.findall(r'\d', text)
    return ''.join(digits)


def recognize_with_paddleocr(ocr, image):
    try:
        result = ocr.predict(image)

        if result and isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'rec_texts' in result[0]:
                rec_texts = result[0].get('rec_texts', [])
                if rec_texts:
                    recognized_text = ' '.join(rec_texts)
                    return extract_digits(recognized_text)
        return ""
    except Exception as e:
        return ""


def preprocess_image(img, method_choice):
    # Для метода 0 - вообще никакой предобработки
    if method_choice == 0:
        return img.copy()

    processed = img.copy()

    if method_choice == 1:
        # Только нормализация
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)

    elif method_choice == 2:
        # Нормализация + дилатация (эллипс)
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.dilate(processed, kernel, iterations=1)

    elif method_choice == 3:
        # Морфологический градиент + дилатация
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(processed, cv2.MORPH_GRADIENT, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.dilate(processed, kernel, iterations=1)

    elif method_choice == 4:
        # Градиент (5x5) + дилатация
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        processed = cv2.morphologyEx(processed, cv2.MORPH_GRADIENT, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.dilate(processed, kernel, iterations=1)

    elif method_choice == 5:
        # Медианный фильтр
        processed = cv2.medianBlur(processed, 3)

    elif method_choice == 6:
        # Медианный фильтр + градиент + дилатация
        processed = cv2.medianBlur(processed, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(processed, cv2.MORPH_GRADIENT, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.dilate(processed, kernel, iterations=1)

    elif method_choice == 7:
        # Top-hat + Black-hat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        tophat = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(processed, cv2.MORPH_BLACKHAT, kernel)
        processed = cv2.add(processed, tophat)
        processed = cv2.subtract(processed, blackhat)

    elif method_choice == 8:
        # Авто-инверсия (анализ центральной зоны)
        h, w = processed.shape
        center = processed[h // 4:3 * h // 4, w // 4:3 * w // 4]
        center_mean = np.mean(center)
        if center_mean > 127:
            processed = cv2.bitwise_not(processed)

    elif method_choice == 9:
        # Гамма-коррекция + нормализация
        processed = apply_gamma(processed, gamma=1.5)
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)

    elif method_choice == 10:
        # Гамма + нормализация + градиент + дилатация
        processed = apply_gamma(processed, gamma=1.5)
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(processed, cv2.MORPH_GRADIENT, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.dilate(processed, kernel, iterations=1)

    return processed


def calculate_accuracy(recognized, target="23"):
    return 100.0 if recognized == target else 0.0


def test_method(ocr, folder_path, method_choice, method_name, show_details=False):
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) +
                         glob.glob(os.path.join(folder_path, "*.JPG")) +
                         glob.glob(os.path.join(folder_path, "*.png")) +
                         glob.glob(os.path.join(folder_path, "*.PNG")))

    if not image_paths:
        return None, 0, 0, 0, []

    total_accuracy = 0.0
    correct_count = 0
    results_table = []
    start_time = time.time()

    for idx, path in enumerate(image_paths, 1):
        filename = os.path.basename(path)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        filtered_img = preprocess_image(img, method_choice)

        # Конвертируем в RGB для PaddleOCR
        processed_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)

        recognized_digits = recognize_with_paddleocr(ocr, processed_rgb)

        accuracy = calculate_accuracy(recognized_digits, target="23")
        total_accuracy += accuracy

        is_correct = (recognized_digits == "23")
        if is_correct:
            correct_count += 1

        results_table.append((filename, recognized_digits, accuracy, is_correct))

        if show_details:
            status = "+" if is_correct else "-"
            print(f"    [{idx:2d}/{len(image_paths)}] {status} {filename[:30]:30} -> '{recognized_digits:5}'")

    elapsed_time = time.time() - start_time
    mean_accuracy = total_accuracy / len(results_table) if results_table else 0

    return results_table, mean_accuracy, correct_count, elapsed_time, len(results_table)


def print_table(data, headers):
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    col_widths = [w + 2 for w in col_widths]

    print("+" + "+".join("-" * w for w in col_widths) + "+")
    header_line = "|"
    for i, header in enumerate(headers):
        header_line += f" {header:<{col_widths[i] - 1}}|"
    print(header_line)
    print("+" + "+".join("-" * w for w in col_widths) + "+")

    for row in data:
        data_line = "|"
        for i, cell in enumerate(row):
            data_line += f" {str(cell):<{col_widths[i] - 1}}|"
        print(data_line)

    print("+" + "+".join("-" * w for w in col_widths) + "+")


def main():
    # ==================================
    folder_path = "year1"
    # ==================================

    print("=" * 80)
    print("TESTING METHODS FOR RECOGNIZING NUMBER 23")
    print("=" * 80)
    print(f"Folder: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"\nERROR: Folder '{folder_path}' not found!")
        return

    image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(folder_path, "*.JPG")) + \
                  glob.glob(os.path.join(folder_path, "*.png")) + \
                  glob.glob(os.path.join(folder_path, "*.PNG"))

    if not image_paths:
        print(f"\nERROR: No images in folder '{folder_path}'!")
        return

    print(f"Images found: {len(image_paths)}")

    print("\nInitializing PaddleOCR...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print("PaddleOCR ready")

    methods = {
        0: "NO FILTERS - Original image only (baseline)",
        1: "Normalization only",
        2: "Normalization + Ellipse dilation",
        3: "Morphological gradient + Dilation",
        4: "Gradient (5x5) + Dilation",
        5: "Median blur (3) only",
        6: "Median blur (3) + Gradient + Dilation",
        7: "Top-hat + Black-hat",
        8: "Auto-inversion by center zone",
        9: "Gamma (1.5) + Normalization",
        10: "Gamma (1.5) + Normalization + Gradient + Dilation"
    }

    # ВЫБОР МЕТОДА
    print("\n" + "=" * 80)
    print("AVAILABLE METHODS:")
    print("=" * 80)
    for num, name in methods.items():
        print(f"  {num:2d}. {name}")
    print(f"  11. RUN ALL METHODS FOR COMPARISON")

    while True:
        try:
            choice = int(input("\nSelect method (0-11): "))
            if 0 <= choice <= 11:
                break
        except:
            print("Please enter number 0-11")

    show_details = input("\nShow details? (y/n): ").lower() == 'y'

    all_results = []

    # ЕСЛИ ВЫБРАН ОТДЕЛЬНЫЙ МЕТОД
    if choice != 11:
        method_name = methods[choice]
        print("\n" + "=" * 80)
        print(f"TESTING METHOD {choice}: {method_name}")
        print("=" * 80)

        results_table, mean_accuracy, correct_count, elapsed_time, total_images = test_method(
            ocr, folder_path, choice, method_name, show_details
        )

        if results_table:
            print(f"\n  Results:")
            print(f"     Correct: {correct_count}/{total_images}")
            print(f"     Accuracy: {mean_accuracy:.2f}%")
            print(f"     Time: {elapsed_time:.2f} sec")

    # ЕСЛИ ВЫБРАН ЗАПУСК ВСЕХ МЕТОДОВ
    else:
        print("\n" + "=" * 80)
        print("STARTING TESTS OF ALL METHODS")
        print("=" * 80)

        for method_num in range(0, 11):
            method_name = methods[method_num]

            print(f"\n[{method_num}/10] {method_name}")
            print("-" * 50)

            results_table, mean_accuracy, correct_count, elapsed_time, total_images = test_method(
                ocr, folder_path, method_num, method_name, show_details
            )

            if results_table is None:
                continue

            print(f"\n  Results:")
            print(f"     Correct: {correct_count}/{total_images}")
            print(f"     Accuracy: {mean_accuracy:.2f}%")
            print(f"     Time: {elapsed_time:.2f} sec")

            all_results.append({
                'num': method_num,
                'name': method_name,
                'correct': correct_count,
                'total': total_images,
                'accuracy': mean_accuracy,
                'time': elapsed_time,
            })

        # Показываем итоговую таблицу только если тестировали все методы
        if all_results:
            all_results.sort(key=lambda x: x['accuracy'], reverse=True)

            print("\n" + "=" * 80)
            print("FINAL RESULTS")
            print("=" * 80)

            table_data = []
            for rank, res in enumerate(all_results, 1):
                table_data.append([
                    f"#{rank}",
                    res['name'][:45],
                    f"{res['correct']}/{res['total']}",
                    f"{res['accuracy']:.1f}%",
                    f"{res['time']:.1f}"
                ])

            headers = ["Rank", "Method Name", "Correct/Total", "Accuracy", "Time (sec)"]
            print_table(table_data, headers)

            best = all_results[0]
            print("\n" + "=" * 80)
            print(f"BEST METHOD: {best['name']}")
            print("=" * 80)
            print(f"   Correct: {best['correct']}/{best['total']}")
            print(f"   Accuracy: {best['accuracy']:.2f}%")
            print(f"   Time: {best['time']:.2f} sec")

    print("\nTesting completed!")


if __name__ == "__main__":
    main()
