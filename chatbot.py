import torch

class ChatBot:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_response(self, user_input, max_length=128):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(user_input, max_length=max_length)["input_ids"].to(self.device)
            input_ids = input_ids.squeeze(0)
            input_tensor = input_ids.unsqueeze(0)

            output = self.model(input_tensor, input_tensor)
            predicted_ids = output.argmax(dim=-1)[0]
            return self.tokenizer.decode(predicted_ids)

    def chat(self):
        print("🧠 Chatbot is ready. Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() == "exit":
                print("👋 Goodbye!")
                break
            response = self.generate_response(user_input)
            print("Bot:", response)
