import mss
import numpy as np
import cv2

with mss.mss() as sct:
    print("Available monitors:", sct.monitors)
    monitor = {"top": 225, "left": 600, "width": 700, "height": 165}

    while True:
        # Capture full screen for context (primary monitor)
        full_screen = np.array(sct.grab(sct.monitors[1]))

        # Draw the mss region rectangle on full screen (in red)
        top_left = (monitor["left"], monitor["top"])
        bottom_right = (monitor["left"] + monitor["width"], monitor["top"] + monitor["height"])
        cv2.rectangle(full_screen, top_left, bottom_right, (0, 0, 255), 2)

        # Capture only the mss region
        region = np.array(sct.grab(monitor))

        # Show both windows side-by-side
        cv2.imshow("Full Screen with Region Highlight", full_screen)
        cv2.imshow("Captured Region Only", region)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
