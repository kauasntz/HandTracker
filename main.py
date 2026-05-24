import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def process_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=process_result
)

def draw_landmarks(frame, result):
    if result is None:
        return frame
    for hand_landmarks in result.hand_landmarks:
        for landmark in hand_landmarks:
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        for connection in connections:
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]
            h, w, _ = frame.shape
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    with HandLandmarker.create_from_options(options) as landmarker:
        timestamp = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            landmarker.detect_async(mp_image, timestamp)
            timestamp += 1

            frame = draw_landmarks(frame, latest_result)

            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()