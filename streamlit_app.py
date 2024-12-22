from app import app
import sys
sys.path.insert(0, "app")
sys.path.insert("app/models")
sys.path.insert("app/services")
sys.path.insert("app/utils")

if __name__ == "__main__":
    app.main()