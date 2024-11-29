# from flask import Flask

# # Application Factory
# def create_app():
#     app = Flask(__name__)

#     # Import routes
#     from app.routes import init_routes
#     init_routes(app)

#     return app


from flask import Flask
from flask_cors import CORS  # Import CORS

def create_app():
    app = Flask(__name__)

    # Enable CORS for all routes
    CORS(app)

    from app.routes import init_routes
    init_routes(app)

    return app
