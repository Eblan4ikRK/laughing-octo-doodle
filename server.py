from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import base64
import cv2
import numpy as np
from collections import Counter
import ddddocr
from PIL import Image, ImageDraw


class NoiseRemover:
    @staticmethod
    def remove_all_noise(img):
        # Проверяем, является ли изображение пустым
        if img is None or img.size == 0:
            raise ValueError("Пустое изображение передано на обработку.")

        # Если изображение в градациях серого, конвертируем его в BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) < 3 or img.shape[2] < 3:
            raise ValueError(f"Некорректное количество каналов в изображении: {img.shape}")

        # Возвращаем изображение
        return img


class CaptchaProcessor:
    @staticmethod
    def keep_most_frequent_colors(image, fill_color=(255, 0, 0)):  # Красный цвет по умолчанию
        center_width = image.width // 4
        center_height = image.height // 4
        center_x = image.width // 2 - center_width // 2
        center_y = image.height // 2 - center_height // 2

        # Извлечение цветов и подсчет частоты
        color_counter = Counter()
        for y in range(center_y, center_y + center_height):
            for x in range(center_x, center_x + center_width):
                color = image.getpixel((x, y))
                if color != (255, 255, 255):  # Исключение белого цвета
                    color_counter[color] += 1

        # Найти самый частый цвет
        most_frequent_color = color_counter.most_common(1)[0][0] if color_counter else (255, 255, 255)

        # Создаем новое изображение с белым фоном
        result_image = Image.new('RGB', image.size, (255, 255, 255))
        draw = ImageDraw.Draw(result_image)

        # Создаем маску для заливки
        mask = Image.new('L', image.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        for y in range(image.height):
            for x in range(image.width):
                color = image.getpixel((x, y))
                if color == most_frequent_color:
                    mask_draw.point((x, y), fill=255)

        # Заливаем область капчи выбранным цветом
        draw.bitmap((0, 0), mask, fill=fill_color)

        return result_image
    
    @staticmethod
    def remove_small_objects(binary_image, min_size=25):
        # Маркировка связных компонентов на бинаризированном изображении
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        # Создание нового изображения для результата
        cleaned_image = np.zeros_like(binary_image)

        # Оставляем только те компоненты, размер которых превышает min_size
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_image[labels == i] = 255

        return cleaned_image

    @staticmethod
    def remove_thin_lines(image):
        # Применение адаптивной бинаризации
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2)

        # Определение структуры ядра для операции "black-hat"
        kernel = np.ones((3, 3), np.uint8)
        
        # Применение операции "black-hat" для удаления тонких линий
        blackhat = cv2.morphologyEx(binary_image, cv2.MORPH_BLACKHAT, kernel)
        
        # Очищаем тонкие линии, вычитая blackhat результат из исходного изображения
        result = cv2.subtract(binary_image, blackhat)
        
        # Удаляем мелкие объекты и остатки линий
        result = CaptchaProcessor.remove_small_objects(result, min_size=100)

        return result

    @staticmethod
    def fix_broken_lines(image_path, output_path):
        # Загрузка изображения
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Не удалось загрузить изображение.")

        # Применение адаптивной бинаризации для более точного выделения контуров
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2)

        # Удаление тонких линий
        binary_image = CaptchaProcessor.remove_thin_lines(image)

        # Удаление мелких объектов на изображении
        binary_image = CaptchaProcessor.remove_small_objects(binary_image, min_size=32)

        # Использование морфологических операций для очистки изображения
        kernel = np.ones((3, 3), np.uint8)

        # Расширение и сужение изображений для удаления тонких линий
        eroded_image = cv2.erode(binary_image, kernel, iterations=1)
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
        
        # Применение морфологической операции закрытия для удаления разрывов
        closed_image = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel, iterations=5)
        
        # Еще одно расширение и сужение для улучшения контуров
        cleaned_image = cv2.erode(closed_image, kernel, iterations=1)
        final_image = cv2.dilate(cleaned_image, kernel, iterations=1)

        # Инверсия цветов, чтобы вернуть изображение в нормальное состояние
        final_image = cv2.bitwise_not(final_image)

        # Сохранение результата
        cv2.imwrite(output_path, final_image)
        


class MyHTTPRequestHandler(BaseHTTPRequestHandler):
    ocr = ddddocr.DdddOcr()

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        logging.info("Received POST data of length: %d", content_length)

        img_data = post_data.decode('utf-8').encode()
        content = base64.b64decode(img_data)

        with open('image.png', 'wb') as fw:
            fw.write(content)

        # Загрузка изображения с использованием OpenCV
        image_path = "image.png"
        image = cv2.imread(image_path)

        # Удаление шума
        processed_image = NoiseRemover.remove_all_noise(image)

        # Конвертируем в PIL для работы с цветами
        processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # Сохраняем только чаще всего используемые цвета и заливаем красным
        result_image = CaptchaProcessor.keep_most_frequent_colors(processed_image_pil, fill_color=(255, 0, 0))

        # Сохранение результата в промежуточный файл
        intermediate_output_path = "intermediate_image.png"
        result_image.save(intermediate_output_path)

        # Исправление прерванных линий
        final_output_path = "final_image.png"
        CaptchaProcessor.fix_broken_lines(intermediate_output_path, final_output_path)

        # Чтение обработанного изображения для OCR
        with open(final_output_path, "rb") as img_file:
            img_data = img_file.read()

        result = self.ocr.classification(img_data)
        logging.info("OCR Result: %s", result)

        self._set_response()
        self.wfile.write(("OK|" + result).encode('utf-8'))
    
        

def run(server_class=HTTPServer, handler_class=MyHTTPRequestHandler, port=80):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    run(port=80)