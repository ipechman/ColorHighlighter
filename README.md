[![License](https://img.shields.io/badge/Code_License-MIT-blue.svg)](LICENSE)

# ColorHighlighter

<p align="center">
  <img src="./pics/color_highlight.jpg" width="100%">
</p>

ColorHighlighter was inspired by the `#FF06B5` color mystery in Cyberpunk 2077. This tool captures a chosen window's content, highlighting this specific color in real-time, aiding fans in observing its presence in-game or elsewhere.

## Requirements

- numpy
- numba
- mss
- pygetwindow
- tkinter
- PIL (Pillow)

Install them using pip:

```bash
pip install numpy numba mss pygetwindow Pillow
```
## Usage
Run the script with an optional hex color argument:

```bash
python ColorHighlighter.py #FF06B5
```
If no color is provided, the default color `#FF06B5` will be used. Upon running, you'll be presented with a list of active windows. Choose a window by entering the respective number. A new window will pop up, displaying the chosen window's content with the specified color highlighted. To exit, press `Ctrl+C` in the terminal or close the tkinter window directly.

## License
[MIT](LICENSE)
