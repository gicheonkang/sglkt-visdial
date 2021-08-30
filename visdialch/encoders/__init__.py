from visdialch.encoders.sgln import SGLN
#from visdialch.encoders.net_utils import AttFlat

def Encoder(model_config, *args):
    name_enc_map = {
    	"sgln": SGLN
    }
    return name_enc_map[model_config["encoder"]](model_config, *args)
