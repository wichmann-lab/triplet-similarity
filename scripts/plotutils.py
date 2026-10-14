import xml.etree.ElementTree as ET
import matplotlib
from matplotlib.markers import MarkerStyle

## install with: pip install svgpath2mpl
from svgpath2mpl import parse_path

# function that parses the path in the SVG files:
def svg_to_marker(svg_path, scale=1.0, center=True):
    # Load and parse SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Find the first path element
    path_elem = root.find(".//{http://www.w3.org/2000/svg}path")
    if path_elem is None or 'd' not in path_elem.attrib:
        raise ValueError("No valid <path d='...'> found in SVG")

    d = path_elem.attrib['d']

    # Parse path using svgpath2mpl
    marker_path = parse_path(d)

    # Optionally centre the marker around (0,0)
    if center:
        marker_path.vertices -= marker_path.vertices.mean(axis=0)

    # Optionally scale
    marker_path.vertices *= scale
    marker_path.vertices[:, 1] *= -1  # Flip Y axis

    return MarkerStyle(marker_path)



tax_colour_dict = {'bee': '#f9de23ff', 
                   'branch': 'sienna',
                   'katydid': '#08da00ff',
                   'ladybug': '#ff1d1dff',
                   'leaf': '#08da00ff',
                   'phyllium': '#04aa00ff',
                   'stick_insect': '#bd9b75ff', 
                   'tree': '#bd9b75ff', 
                   'tree_trunk':  '#bd9b75ff',
                   'wasp':'#ff8c1dff'}

action_colour_dict = {'coin_3': '#00b329ff',
                      'coin_2': '#00c930ff',
                      'coin_1': '#00e639ff',
                      'key_4': '#be6e00ff',
                      'key_3': '#f49900ff',
                      'key_1': '#ffb871ff',
                      'screwdriver_1': '#ffb3b3ff',
                      'screwdriver_5': '#dc0000ff',
                      'screwdriver_6': '#b30000ff',
                      'screwdriver_7': '#8f0000ff',
                      'matches_1': '#deb7ffff',
                      'matches_6': '#8d00c2ff',
                      'matches_8': '#71009cff',
                      'letter_opener_1': '#a7c8ffff',  
                     }

