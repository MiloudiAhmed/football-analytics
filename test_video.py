import cv2

print("START")

cap = cv2.VideoCapture("input_videos/vedio_test.mp4")

if not cap.isOpened():
    print("Erreur ouverture vidéo")
else:
    print("Vidéo ouverte")

ret, frame = cap.read()

if ret:
    print("Première frame lue")
else:
    print("Erreur lecture frame")

cap.release()

print("END")

