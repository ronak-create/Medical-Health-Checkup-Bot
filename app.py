import google.generativeai as genai
import os
from PIL import Image
import datetime

# Configure API Key
genai.configure(api_key="AIzaSyAvswdOPVKHM1ew374flcLM_RHnpJFji6A")  # Replace with your actual API key

# Initialize the model (Gemini-1.5-Flash)
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to process text input
def process_text_input(user_input):
    response = model.generate_content(user_input)
    return response.text.strip()

# Function to preprocess and analyze two images together
def process_image_input(image_path1, image_path2):
    # Load and preprocess both images
    try:
        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")
        
        # Resize images (you may need to adjust this based on model requirements)
        image1 = image1.resize((224, 224))
        image2 = image2.resize((224, 224))

        # Save images temporarily for analysis if needed
        image1_path_temp = "image1_temp.jpg"
        image2_path_temp = "image2_temp.jpg"
        image1.save(image1_path_temp)
        image2.save(image2_path_temp)

        # Send image paths to the model (assumes the model accepts image inputs via URL or file paths)
        # If model supports images, you may use:
        response = model.generate_content("Combine analysis of these two images:", images=[image1_path_temp, image2_path_temp])
        
        # Placeholder: Provide a dummy response for combined image analysis
        combined_analysis = f"Images {image_path1} and {image_path2} analyzed together."
        return combined_analysis
    
    except Exception as e:
        return f"Error processing images: {e}"

# Main function for chatbot
def chat():
    print("Welcome to the Gemini Chatbot!")
    print("Type 'exit' to end the conversation.\n")
    print("Type 'image' to input two images for combined analysis.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "image":
            image_path1 = "image1.jpg"
            image_path2 = "image2.jpg"
            
            if os.path.exists(image_path1) and os.path.exists(image_path2):
                bot_response = process_image_input(image_path1, image_path2)
            else:
                bot_response = "Error: One or both image files not found."
        else:
            bot_response = process_text_input(user_input)
        
        print(f"Bot: {bot_response}")

# Run the chatbot
if __name__ == '__main__':
    chat()
