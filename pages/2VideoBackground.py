import streamlit as st
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from rembg import remove
from PIL import Image, ImageDraw, ImageFont
import os

def save_uploaded_file(uploaded_file, filename, save_dir="./static/video"):
file_path = os.path.join(save_dir, filename)
with open(file_path, "wb") as f:
f.write(uploaded_file.getbuffer())
return file_path

def add_text_to_background(frame, text, font_size=24, color=(255, 0, 0), position=(50, 50)):
# Convert frame to PIL image to draw text
pil_img = Image.fromarray(frame)
draw = ImageDraw.Draw(pil_img)

# Load a font (adjust path or use default if needed)
try:
font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
font = ImageFont.load_default() # Fallback if custom font fails to load

# Draw text on image with user-defined properties
draw.text(position, text, fill=color, font=font)

return np.array(pil_img)

def process_video_with_layers(input_path, output_path, text, font_size, color, position):
original_video = VideoFileClip(input_path)
fps = original_video.fps
frames = []

for frame in original_video.iter_frames():
# Step 1: Separate the person (foreground) from the background
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
foreground_rgba = remove(frame_rgb)

# Convert to RGB and create alpha mask
foreground_rgb = cv2.cvtColor(np.array(foreground_rgba), cv2.COLOR_RGBA2RGB)
foreground_alpha = cv2.cvtColor(np.array(foreground_rgba), cv2.COLOR_RGBA2GRAY)

# Step 2: Create a background with text using customization options
background_with_text = add_text_to_background(
frame_rgb, text=text, font_size=font_size, color=color, position=position
)

# Step 3: Combine masked background and foreground
# Mask the background with the alpha channel of the foreground
masked_background = cv2.bitwise_and(background_with_text, background_with_text, mask=cv2.bitwise_not(foreground_alpha))

# Combine with full opacity foreground
combined_frame = cv2.add(foreground_rgb, masked_background)

frames.append(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))

# Save the final frames as a video
result_clip = ImageSequenceClip(frames, fps=fps)
result_clip.write_videofile(output_path, codec="libx264")

# Streamlit interface for video processing
st.set_page_config(page_title="Customizable Text-Behind Video Tool", page_icon="🎥")

if __name__ == "__main__":
st.title("Text-Behind-Person Video Effect")
st.write("Upload a video and customize text properties to add text behind the human in the video.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
text_input = st.text_input("Enter the text to display", value="Your Text Here")

# Text customization options
font_size = st.slider("Font Size", min_value=10, max_value=100, value=24)
color = st.color_picker("Text Color", value="#FF0000")
color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) # Convert hex to RGB
x_offset = st.number_input("X Offset Position", min_value=0, max_value=1920, value=50)
y_offset = st.number_input("Y Offset Position", min_value=0, max_value=1080, value=50)

if uploaded_file is not None and text_input:
video_dir = "./output"
os.makedirs(video_dir, exist_ok=True)
file_name = "uploaded_video.mp4"
video_path = save_uploaded_file(uploaded_file, file_name)
output_path = os.path.join(video_dir, "output_with_text_behind_person.mp4")

st.video(data=uploaded_file)
st.success(f"Video uploaded as {video_path}")

if st.button("Process Video"):
with st.spinner("Processing..."):
process_video_with_layers(
video_path, output_path, text=text_input, font_size=font_size,
color=color_rgb, position=(x_offset, y_offset)
)
st.success("Video processed successfully!")
st.video(output_path)
else:
st.warning("Please upload a video file and enter text to start.")
