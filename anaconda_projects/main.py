from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello AI Engineer"}

@app.get("/students")
def get_students():
    return ["Rahul", "Shubham"]