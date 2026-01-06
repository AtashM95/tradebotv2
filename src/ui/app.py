
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .routes import router

app = FastAPI()
app.include_router(router)
app.mount('/static', StaticFiles(directory='src/ui/static'), name='static')

templates = Jinja2Templates(directory='src/ui/templates')
