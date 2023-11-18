from huggingface_hub import InferenceClient

hist = "user1: hey\nuser2: wassup\nuser1: not much, hbu?\n"
inference = InferenceClient()


def prompt(history, user=None):
    result = inference.text_generation(history, model="HuggingFaceH4/zephyr-7b-beta")
    result = result[: result.find("\n")]
    if user and result.find(user) != -1:
        return result[len(user) + 2:]
    else:
        return False


print("response:", prompt(hist, user="user2"))
