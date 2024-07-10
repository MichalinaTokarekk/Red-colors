import cv2
import numpy as np

# Rozpoczęcie przechwytywania obrazu z kamery
cap = cv2.VideoCapture(0)

while True:
    # Przechwyć ramkę
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja BGR na HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Zakres kolorów czerwonych w HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Stworzenie maski dla czerwonego koloru
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Znajdź kontury w masce
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Wybierz największy kontur
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Wyświetlanie ramki i maski
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Wyjście przez naciśnięcie 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie przechwytywania i zamknięcie wszystkich okien
cap.release()
cv2.destroyAllWindows()
