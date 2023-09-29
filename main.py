import numpy as np
from numba import cuda, float32
from mss import mss
import pygetwindow as gw
import tkinter as tk
from PIL import Image, ImageTk
import time
import sys

# CUDA kernel to highlight a specific color in the image.
@cuda.jit
def highlight_color_kernel(img, target_rgb, threshold, output):
    x, y = cuda.grid(2)
    if x < img.shape[0] and y < img.shape[1]:
        r, g, b = img[x, y]
        distance = ((r - target_rgb[0]) ** 2 + (g - target_rgb[1]) ** 2 + (b - target_rgb[2]) ** 2) ** 0.5

        if distance <= threshold:
            scale_factor = distance / threshold
            intensity = int(255 * scale_factor)
            output[x, y, 0] = intensity
            output[x, y, 1] = intensity
            output[x, y, 2] = intensity
        else:
            output[x, y, 0] = 255
            output[x, y, 1] = 255
            output[x, y, 2] = 255


def highlight_color_with_numba(img, hex_color, threshold=100):
    img_np = np.array(img)
    hex_color = hex_color.lstrip('#')
    target_rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)
    output_np = np.zeros_like(img_np)

    d_img = cuda.to_device(img_np)
    d_output = cuda.device_array_like(img_np)
    d_target_rgb = cuda.to_device(target_rgb)

    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(img_np.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(img_np.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    highlight_color_kernel[blocks_per_grid, threads_per_block](d_img, d_target_rgb, threshold, d_output)
    d_output.copy_to_host(output_np)

    return Image.fromarray(output_np.astype(np.uint8))


def capture_screenshot(window):
    with mss() as sct:
        region = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
        sct_img = sct.grab(region)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def create_display_window(title="Highlighted Window"):
    root = tk.Tk()
    root.title(title)
    label = tk.Label(root)
    label.pack()
    return root, label


def update_display_window(window, label, image):
    tk_image = ImageTk.PhotoImage(image=image)
    label.config(image=tk_image)
    label.image = tk_image
    window.update()


def get_window_title_choice():
    windows = gw.getAllWindows()
    window_titles = [win.title for win in windows if win.visible and win.title]

    if not window_titles:
        print("No visible windows found.")
        return None

    for i, title in enumerate(window_titles):
        print(f"{i + 1}. {title}")

    try:
        choice = int(input("Choose a window by number: ")) - 1
        if choice < 0 or choice >= len(window_titles):
            print("Invalid choice.")
            return None
        return window_titles[choice]
    except ValueError:
        print("Please enter a valid number.")
        return None


def main(hex_color):
    target_window_title = get_window_title_choice()
    if not target_window_title:
        return

    # Ensure that the window with the given title exists.
    matching_windows = gw.getWindowsWithTitle(target_window_title)
    if not matching_windows:
        print(f"No window with title '{target_window_title}' found.")
        return

    window = matching_windows[0]

    fps = 30.0
    capture_delay = 1.0 / fps

    display_window, display_label = create_display_window()

    try:
        while True:
            start_time = time.time()

            screenshot = capture_screenshot(window)
            highlighted_img = highlight_color_with_numba(screenshot, hex_color)
            highlighted_img_resized = highlighted_img.resize((int(window.width * 0.75), int(window.height * 0.75)))

            update_display_window(display_window, display_label, highlighted_img_resized)

            elapsed_time = time.time() - start_time
            time_to_sleep = capture_delay - elapsed_time
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
    except KeyboardInterrupt:
        display_window.destroy()
    except tk.TclError:  # To handle cases where the tkinter window is closed directly.
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:  # Allow hex color to be passed as an argument.
        hex_color = sys.argv[1]
    else:
        hex_color = "#FF06B5"
    main(hex_color)
