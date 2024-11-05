# Image Color Extraction Tool

![Test Image](https://github.com/alibahrami2001/image_color_extraction/blob/main/image/test.png)

# Test the Jupyter Notebook Online

Click the button below to launch the Jupyter Notebook in Google Colab:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alibahrami2001/image_color_extraction/blob/main/devbook.ipynb)

## Overview
The Image Color Extraction Tool allows users to extract and visualize dominant colors from images. It utilizes K-Means clustering to identify and cluster colors for analysis. This tool is designed for artists, designers, and anyone interested in color analysis.

## Features
- Extracts dominant colors from uploaded images.
- Clusters colors using K-Means for better visualization.
- Interactive web interface powered by Gradio.
- Configurable parameters to adjust the number of colors extracted.

![Image](https://github.com/alibahrami2001/image_color_extraction/blob/main/image/acbf4be6-6a5d-4eec-b771-6cafdfbb67f3.png)
## Requirements
To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `scikit-learn`
- `gradio`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib Pillow scikit-learn gradio
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/alibahrami2001/image-color-extraction.git
   cd image-color-extraction
   ```

2. Run the Gradio interface:
   ```bash
   python your_script_name.py
   ```

3. Open the provided URL in your web browser to access the tool.

## How to Use
1. Upload an image by clicking on the "Input Image" area.
2. Adjust the "Max Number of Colors" slider to specify how many dominant colors you want to extract.
3. Click the "Run" button to extract colors and visualize the results.
4. The extracted colors will be displayed as an image and in a table format.

## Example
Watch a demonstration of the Image Color Extraction Tool in action:

![Animation](https://github.com/alibahrami2001/image_color_extraction/blob/main/image/Animation.gif)

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the LICENSE file for more details.

## Acknowledgments
```

Feel free to modify any sections, add examples, or include additional details that you think would be helpful for users!
