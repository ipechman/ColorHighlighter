import numpy as np
from numba import cuda
import tkinter as tk
from PIL import Image, ImageTk
import sys
import win32gui
import win32ui
import win32con
import win32api
import cv2


def hex_to_hsv(hex_color):
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]

    # Convert RGB to BGR since OpenCV uses BGR format
    bgr = np.uint8([[[b, g, r]]])

    # Convert BGR to HSV using OpenCV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]

    h, s, v = hsv
    return h, s, v


# CUDA kernel to highlight a specific color in the image.
@cuda.jit
def optimized_highlight_color_kernel(img, Th, Ts, output):
    y, x = cuda.grid(2)
    if y < img.shape[0] and x < img.shape[1]:
        # H 0-179
        # S 0-255
        # V 0-255
        h = img[y, x, 0]
        s = img[y, x, 1]
        v = img[y, x, 2]

        # Calculate distance in HUE considering wrap-around
        hue_diff = min(abs(h - Th), 180 - abs(h - Th))
        # Calculate distance in SATURATION
        sat_diff = abs(s - Ts)

        # Normalize differences
        hue_diff_normalized = hue_diff / 180.0  # as the range for H is 0-179
        sat_diff_normalized = sat_diff / 255.0  # as the range for S is 0-255

        # Combined difference metric
        combined_diff = (hue_diff_normalized + sat_diff_normalized) / 2.0

        # Clip combined difference to be within threshold
        threshold = 0.05
        combined_diff = min(max(combined_diff, 0), threshold)

        # Compute gradient value based on the clipped differences
        gradient_value = 255 * (1 - combined_diff / threshold)
        gradient_value = int(gradient_value)

        # If brightness is too low or combined difference is at the maximum, keep it black, else apply gradient
        output[y, x] = (
        gradient_value, gradient_value, gradient_value) if v >= 64 and combined_diff < threshold else (0, 0, 0)


def optimized_highlight_color_with_numba(img_hsv, Th, Ts):
    img_np = np.array(img_hsv, dtype=np.uint8)
    output_np = np.zeros_like(img_np)

    d_img = cuda.to_device(img_np)
    d_output = cuda.device_array_like(img_np)

    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(img_np.shape[1] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(img_np.shape[0] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    optimized_highlight_color_kernel[blocks_per_grid, threads_per_block](d_img, Th, Ts, d_output)
    d_output.copy_to_host(output_np)

    return Image.fromarray(output_np)


def capture_screenshot(debug=True):
    """
    Captures a screenshot of the primary display and returns it in HSV format.
    If debug is True, the function also saves a BGR image and prints some debug information.
    """
    # Get primary display dimensions
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

    # Capture the screenshot using WinAPI
    hDC = win32gui.GetDC(0)
    dcObj = win32ui.CreateDCFromHandle(hDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, screen_width, screen_height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (screen_width, screen_height), dcObj, (0, 0), win32con.SRCCOPY)

    # Retrieve and reshape the bitmap data
    bmpstr = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8)
    img.shape = (screen_height, screen_width, -1)  # Auto infer the last dimension
    if img.shape[2] == 4:
        img = img[:, :, :3]  # Drop the fourth channel if present

    # Debugging information and save BGR screenshot
    if debug:
        cv2.imwrite("screenshot_debug_bgr.png", img)
        print(f"img_bgr dtype: {img.dtype}, shape: {img.shape}")

    # Convert from BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Clean up
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(0, hDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return img_hsv


def create_display_window(title="Color Highlighter"):
    root = tk.Tk()
    root.title(title)
    root.configure(bg='black')  # Set the background color of the root window to black
    label = tk.Label(root, bg='black')  # Set the background color of the label to black
    label.pack()
    return root, label


def update_display_window(window, label, image):
    tk_image = ImageTk.PhotoImage(image=image)
    label.config(image=tk_image)
    label.image = tk_image
    window.update()


def main(hex_color):
    h, s, v = hex_to_hsv(hex_color)

    display_window, display_label = create_display_window()

    # H 0-179
    # S 0-255
    # V 0-255
    # Define aspect ratio
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    aspect_ratio = screen_width / screen_height

    screenshot = capture_screenshot(debug=False)

    while True:
        # Main loop
        screenshot = capture_screenshot(debug=False)
        highlighted_img = optimized_highlight_color_with_numba(screenshot, h, s)
        highlighted_img_resized = highlighted_img.resize(
            (int(screen_width * 0.5), int(screen_width / aspect_ratio * 0.5)))
        update_display_window(display_window, display_label, highlighted_img_resized)


if __name__ == "__main__":
    if len(sys.argv) > 1:  # Allow hex color to be passed as an argument.
        hex_color = sys.argv[1]
    else:
        hex_color = "#FF06B5"
    main(hex_color)
