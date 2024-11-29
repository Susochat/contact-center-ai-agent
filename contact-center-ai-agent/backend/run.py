from app import create_app

# Create the Flask app
app = create_app()

# Run the app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=True)
