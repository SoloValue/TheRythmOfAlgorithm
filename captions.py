## It's just a file that helps captioning images ##
## === it can then be ignored/deleted == ##

from PIL import Image
from PIL import ImageDraw


# Open the image
image1 = Image.open("Models running results.png")
# Define text
text1 = "Models run results"
# Create a draw object
draw = ImageDraw.Draw(image1)
# Calculate text1 size
text1_size1 = draw.textsize(text1)
# Calculate text1 position
# Draw text1 on image
draw.text((1.0, 0.5), text1, fill='black')
# Save the new image with the caption
image1.save("Figure 1: Models_running_results_caption.png")

# Open the image
image = Image.open("Queries results.png")
# Define text
text = "Figure 2: Query 0, Query2, Query3 Results"
# Create a draw object
draw = ImageDraw.Draw(image)
# Calculate text size
text_size = draw.textsize(text)
# Calculate text position
x = image.width // 2 
y = image.height - 240 
# Draw text on image
draw.text((x, y), text, fill='black')
# Save the new image with the caption
image.save("queries_results_caption.png")



