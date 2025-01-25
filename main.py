import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Загрузка предварительно обученной модели
model = MobileNetV2(weights='imagenet')

def identify_plant(frame):
    """Идентификация растения из кадра."""
    # Изменение размера изображения до 224x224 (размер входного изображения для MobileNetV2)
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(resized_frame, axis=0)
    img_array = preprocess_input(img_array)

    # Прогнозирование
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Берем топ-3 результата

    return decoded_predictions

def main():
    # Подключение камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Камера не обнаружена.")
        return

    print("Запущен сканер растений. Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения с камеры.")
            break

        # Анализируем кадр каждые 30 кадров (примерно 1 раз в секунду при 30 fps)
        predictions = identify_plant(frame)
        for i, (imagenet_id, label, prob) in enumerate(predictions):
            text = f"{label}: {prob * 100:.2f}%"
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Отображение кадра с результатами
        cv2.imshow('Сканер растений', frame)

        # Нажмите 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
