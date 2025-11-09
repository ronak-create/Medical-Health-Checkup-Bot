import openai

# Set your OpenAI API key
openai.api_key = "OPEN_AI_KEY"

def chat_with_gpt():
    print("Chatbot is running! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        try:    
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}]
            )
            reply = response["choices"][0]["message"]["content"]
            print("Chatbot:", reply)
        except Exception as e:
            print("Error:", str(e))

if __name__ == "__main__":
    chat_with_gpt()
