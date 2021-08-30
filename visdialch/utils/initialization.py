import torch

def initialize_model_weights(model, initialization="he", lstm_initialization="he"):
    if initialization == "he":
        print("kaiming normal initialization.")
    elif initialization == "xavier":
        print("xavier normal initialization.")
    else:
        print("default initialization, no changes made.")
    if(initialization):
        for name, param in model.named_parameters():
            # Bias params
            if("bias" in name.split(".")[-1]):
                param.data.zero_()

            # Batchnorm weight params
            elif("weight" in name.split(".")[-1] and len(param.size())==1):
                continue
            # LSTM weight params
            elif("weight" in name.split(".")[-1] and "lstm" in name):
                if "xavier" in lstm_initialization:
                    torch.nn.init.xavier_normal_(param)
                elif "he" in lstm_initialization:
                    torch.nn.init.kaiming_normal_(param)
            # Other weight params
            elif("weight" in name.split(".")[-1] and "lstm" not in name):
                if "xavier" in initialization:
                    torch.nn.init.xavier_normal_(param)
                elif "he" in initialization:
                    torch.nn.init.kaiming_normal_(param)