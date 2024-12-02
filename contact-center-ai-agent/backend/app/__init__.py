
############################################################# FOR DEFAULT MODELs ################################################

from flask import Flask
from flask_cors import CORS  # Import CORS

def create_app():
    app = Flask(__name__)

    # Enable CORS for all routes
    CORS(app)

    # For distilGPT2 model
    # from app.routes_dgpt2 import init_routes

    # For falcon-7b-instruct model
    # from app.routes_fw1b import init_routes

    # For DistilBert and DistilGPT2 combined model
    from app.routes_dbert_dgpt2 import init_routes

    init_routes(app)

    return app



############################################################# FOR FINE TUNED MODEL ################################################

# from flask import Flask
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from flask_cors import CORS  # Import CORS

# def create_app():
#     app = Flask(__name__)

#     # Enable CORS for all routes
#     CORS(app)

#     # Load the fine-tuned model and tokenizer
#     model = GPT2LMHeadModel.from_pretrained('models/fine_tuned_model')
#     tokenizer = GPT2Tokenizer.from_pretrained('models/fine_tuned_model')

#     # Set padding token
#     tokenizer.pad_token = tokenizer.eos_token

#     app.config['MODEL'] = model
#     app.config['TOKENIZER'] = tokenizer

#     return app
