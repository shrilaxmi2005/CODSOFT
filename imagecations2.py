from transformers import BlipProcessor, BlipForConditionalGeneration
from tkinter import Tk, filedialog, Label, Button
from PIL import Image, ImageTk

# Load pre-trained model and processor
processor = BlipProcessor.from_pretrained("salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("salesforce/blip-image-captioning-base")

# Function to generate caption from image
def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Function to upload and display image
def upload_image():
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        caption = generate_caption(file_path)  # Generate caption for selected image
        img = Image.open(file_path).resize((300, 300))  # Resize image to fit the label
        img_tk = ImageTk.PhotoImage(img)  # Convert image to Tkinter compatible format
        image_label.configure(image=img_tk)  # Update the image on the label
        image_label.image = img_tk  # Keep a reference to avoid garbage collection
        caption_label.config(text=f"Caption: {caption}")  # Display caption

# Initialize the Tkinter window
root = Tk()
root.title("Image Caption Generator")
root.geometry("400x500")  # Set window size

# Label for instructions
Label(root, text="Upload an image to generate a caption", font=("Arial", 14)).pack(pady=10)

# Label to display uploaded image
image_label = Label(root)
image_label.pack(pady=10)

# Button to trigger image upload
Button(root, text="Upload Image", command=upload_image).pack(pady=10)

# Label to display the caption generated
caption_label = Label(root, text="", wraplength=350, font=("Arial", 12))  # Wrap text to fit
caption_label.pack(pady=10)

# Start Tkinter main loop
root.mainloop()
