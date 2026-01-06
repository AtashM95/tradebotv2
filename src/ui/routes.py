
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from .app import templates

router = APIRouter()

@router.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('dashboard.html', {'request': request})

@router.get('/watchlist', response_class=HTMLResponse)
async def watchlist(request: Request):
    return templates.TemplateResponse('watchlist.html', {'request': request})
