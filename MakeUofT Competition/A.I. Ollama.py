#Ollama
import ollama

#MiDas Model PyTorch
import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

print("Starting Image-Depth Processing")

#Load Image and Apply Transformers
img = cv2.imread("ESP32-CAM-Images\TestImage2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)    

#Predict and Resize to Original Resolution
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()    
    
plt.imshow(output)
plt.show()    


"""
#Model Setup
model = "makeUofT_guide_model"
personality = "Please be very concise with your replies. Limit them to 1-2 sentences and simple instructions."
response = ollama.create(model=model, from_='llama3.2', system=personality)

#Variables
conversation = []

#Generate Response
def generate_response(model, chat_log):
    return ollama.chat(model=model, messages=chat_log)

#Print Message Line by Line
def print_response(reply, responder = 'Girlfriend', switch = True):
    reply = reply.replace("* ", "*[CUT]").replace(" *", "[CUT]*").replace("! ", "![CUT]").replace("\\", "")
    reply = reply.replace("? ", "?[CUT]").replace("...", "[DCUT]").replace(". ", ".[CUT]").replace("[DCUT]", "...")
    reply_messages = reply.split("[CUT]")
    for line in reply_messages:
        if switch == True:
            print(f"\n{responder}: " + line)
            switch = False
        else:
            print(" "*(len(responder)+2) + line)
            
            
#def create_setting()




#Main Loop
while True:
    #Prompt User Input
    prompt = input("\nYou: ")
    #Log User Input to Chat Logs
    conversation.append({"role": "user", "content": prompt})
    #Generate Response
    response = str(generate_response(model, conversation))
    reply_message = response[response.find("content=")+9:response.rfind(", images")-1]
    #Log A.I. Response
    conversation.append({"role": "assistant", "content": reply_message})
    #Organize and Print Response
    print_response(reply_message, "Guide")
"""

    

