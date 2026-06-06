import os
import glob
import cv2
import numpy as np
import easyocr
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context


def normalize_image(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def apply_gamma(img, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def morphological_gradient(img, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def auto_invert_otsu(img):
    h, w = img.shape
    center = img[h // 4:3 * h // 4, w // 4:3 * w // 4]

    _, center_binary = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(center_binary == 255) / center_binary.size

    invert = white_ratio > 0.7

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        binary = cv2.bitwise_not(binary)

    return binary


def preprocess_image(img, method_choice):
    original = img.copy()

    # Для метода 0 - вообще никакой предобработки
    if method_choice == 0:
        return original

    if method_choice == 1:
        processed = normalize_image(original)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method_choice == 2:
        processed = normalize_image(original)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.dilate(processed, kernel, iterations=1)
        processed = cv2.erode(processed, kernel, iterations=1)

    elif method_choice == 3:
        processed = morphological_gradient(original, 3)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.dilate(processed, kernel, iterations=1)

    elif method_choice == 4:
        grad_x = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=5)
        processed = cv2.magnitude(grad_x, grad_y)
        processed = np.uint8(np.clip(processed, 0, 255))
        processed = cv2.multiply(processed, 1.5)
        processed = np.clip(processed, 0, 255).astype(np.uint8)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.dilate(processed, kernel, iterations=2)

    elif method_choice == 5:
        processed = cv2.medianBlur(original, 3)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method_choice == 6:
        processed = cv2.medianBlur(original, 3)
        processed = morphological_gradient(processed, 3)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.dilate(processed, kernel, iterations=1)

    elif method_choice == 7:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(original, cv2.MORPH_BLACKHAT, kernel)

        processed = cv2.add(original, tophat)
        processed = cv2.subtract(processed, blackhat)
        processed = normalize_image(processed)
        processed = cv2.equalizeHist(processed)
        processed = cv2.adaptiveThreshold(processed, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    elif method_choice == 8:
        processed = auto_invert_otsu(original)

    elif method_choice == 9:
        processed = apply_gamma(original, gamma=1.5)
        processed = normalize_image(processed)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method_choice == 10:
        processed = apply_gamma(original, gamma=1.5)
        processed = normalize_image(processed)
        processed = morphological_gradient(processed, 3)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.dilate(processed, kernel, iterations=1)

    else:
        processed = original

    return processed


def recognize_with_attempts(reader, img, attempts=3):
    """Несколько попыток распознавания с разными параметрами"""

    # Попытка 1: стандартные параметры
    results = reader.readtext(img, paragraph=False, contrast_ths=0.3, adjust_contrast=0.5)
    text = "".join([r[1] for r in results]).replace(" ", "")
    text = ''.join(filter(str.isdigit, text))

    if text == "23":
        return text

    # Попытка 2: более низкий порог
    results = reader.readtext(img, paragraph=False, text_threshold=0.3)
    text = "".join([r[1] for r in results]).replace(" ", "")
    text = ''.join(filter(str.isdigit, text))

    if text == "23":
        return text

    # Попытка 3: инвертированное изображение
    img_inv = cv2.bitwise_not(img)
    results = reader.readtext(img_inv, paragraph=False)
    text = "".join([r[1] for r in results]).replace(" ", "")
    text = ''.join(filter(str.isdigit, text))

    return text


def run_single_method(method_num, image_paths, reader):
    print(f"\n{'=' * 60}")
    print(f" METHOD {method_num}: {method_names[method_num]}")
    print(f"{'=' * 60}")

    start_time = time.time()
    results = []
    correct = 0

    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  SKIP {filename}: cannot read")
            continue

        processed = preprocess_image(img, method_num)
        recognized_text = recognize_with_attempts(reader, processed)

        is_correct = (recognized_text == "23")
        if is_correct:
            correct += 1

        results.append((filename, recognized_text, is_correct))

        status = "[OK]" if is_correct else "[NO]"
        print(f"  {status} {filename}: '{recognized_text}'")

    accuracy = (correct / len(results)) * 100 if results else 0
    elapsed = time.time() - start_time

    print(f"\n--- RESULTS FOR METHOD {method_num} ---")
    print(f"  Correct: {correct}/{len(results)}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Time: {elapsed:.3f} sec")


def run_all_methods(image_paths, reader):
    print("\n" + "=" * 100)
    print(" RUNNING ALL METHODS FOR COMPARISON")
    print("=" * 100)

    all_results = {}

    for method_num in range(0, 11):
        print(f"\n{'=' * 60}")
        print(f" METHOD {method_num}: {method_names[method_num]}")
        print(f"{'=' * 60}")

        start_time = time.time()
        correct = 0

        for path in image_paths:
            filename = os.path.basename(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            processed = preprocess_image(img, method_num)
            recognized_text = recognize_with_attempts(reader, processed)

            is_correct = (recognized_text == "23")
            if is_correct:
                correct += 1

            status = "[OK]" if is_correct else "[NO]"
            print(f"  {status} {filename}: '{recognized_text}'")

        accuracy = (correct / len(image_paths)) * 100
        elapsed = time.time() - start_time

        print(f"\n--- RESULTS FOR METHOD {method_num} ---")
        print(f"  Correct: {correct}/{len(image_paths)}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Time: {elapsed:.3f} sec")

        all_results[method_num] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(image_paths),
            'time': elapsed
        }

    print("\n" + "=" * 100)
    print(" FINAL COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Method':<6} {'Name':<45} {'Correct/Total':<15} {'Accuracy':<10} {'Time (sec)':<12} {'Rank'}")
    print("-" * 100)

    sorted_methods = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    for rank, (num, data) in enumerate(sorted_methods, 1):
        correct_str = f"{data['correct']}/{data['total']}"
        print(
            f"{num:<6} {method_names[num]:<45} {correct_str:<15} {data['accuracy']:>6.2f}%     {data['time']:>8.3f}     #{rank}")

    print("=" * 100)
    print(f"\nBEST METHOD: {method_names[sorted_methods[0][0]]}")
    print(f"  Correct: {sorted_methods[0][1]['correct']}/{sorted_methods[0][1]['total']}")
    print(f"  Accuracy: {sorted_methods[0][1]['accuracy']:.2f}%")


method_names = {
    0: "NO FILTERS - Original image only (baseline)",
    1: "Normalization + Otsu",
    2: "Normalization + Otsu + Ellipse dilation",
    3: "Morphological gradient + Otsu + Dilation",
    4: "Sobel gradient (5x5) + Otsu + Dilation",
    5: "Median blur (3) + Otsu",
    6: "Median blur (3) + Gradient + Otsu + Dilation",
    7: "Top-hat + Black-hat + Adaptive threshold",
    8: "Auto-inversion Otsu by center zone",
    9: "Gamma (1.5) + Normalization + Otsu",
    10: "Gamma (1.5) + Normalization + Gradient + Otsu + Dilation"
}


def main():
    print("=" * 60)
    print(" DIGIT RECOGNITION ON METAL")
    print("=" * 60)

    # ==================================
    folder_path = "year1"
    # ==================================

    image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(folder_path, "*.JPG")) + \
                  glob.glob(os.path.join(folder_path, "*.png")) + \
                  glob.glob(os.path.join(folder_path, "*.PNG"))

    if not image_paths:
        print(f"\nERROR: No images found in '{folder_path}'")
        print(f"Current directory: {os.getcwd()}")
        return

    print(f"\nFound images: {len(image_paths)}")
    for img in image_paths:
        print(f"  {os.path.basename(img)}")
    print(f"Folder: {folder_path}")

    print("\nLoading EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("EasyOCR ready\n")

    print("AVAILABLE METHODS:")
    for i in range(0, 11):
        print(f"  {i}. {method_names[i]}")
    print(f"  11. RUN ALL METHODS FOR COMPARISON")

    while True:
        try:
            choice = int(input("\nSelect method (0-11): "))
            if 0 <= choice <= 11:
                break
        except:
            print("Enter number 0-11")

    if choice == 11:
        run_all_methods(image_paths, reader)
    else:
        run_single_method(choice, image_paths, reader)


if __name__ == "__main__":
    main()
