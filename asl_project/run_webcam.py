import cv2
import time
import hydra
from omegaconf import DictConfig

from src.inference import ASLPredictor


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint_path = cfg.inference.checkpoint_path

    if not checkpoint_path:
        print("ОШИБКА: Не указан путь к чекпоинту в configs/config.yaml (секция inference.checkpoint_path)")
        return

    config_path = "configs/config.yaml"

    print(f"Используем чекпоинт: {checkpoint_path}")

    try:
        predictor = ASLPredictor(cfg)
    except FileNotFoundError:
        print(f"Фатальная ошибка: Файл весов не найден: {checkpoint_path}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    box_size = 300
    x_start, y_start = 100, 100
    x_end, y_end = x_start + box_size, y_start + box_size
    fps_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        roi = frame[y_start:y_end, x_start:x_end]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        label, prob = predictor.predict(roi_rgb)

        if prob > 0.8: color = (0, 255, 0)
        elif prob > 0.5: color = (0, 255, 255)
        else: color = (0, 0, 255)

        cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), color, 3)
        cv2.putText(display_frame, f"{label} ({prob:.1%})", (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        curr_time = time.time()
        fps = 1/(curr_time - fps_time)
        fps_time = curr_time
        cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("ASL System", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
