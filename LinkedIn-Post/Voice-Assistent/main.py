from gradio_client import Client
import pyttsx3 

def SpeakText(command):
     
    # Initialize the engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 175) 
    voices = engine.getProperty('voices')       
    engine.setProperty('voice', voices[1].id)     
    engine.say(command) 
    engine.runAndWait()
     

while True:
    input_question = input("How can I help You? : ")
    if input_question == "exit":
        break
    client = Client("https://cohereforai-c4ai-command-r-plus.hf.space/--replicas/993c3/")
    result = client.predict(
        input_question,
        api_name="/generate_response"
    )
    ans = result[0][1]
    SpeakText(ans)

    