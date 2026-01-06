
from fastapi import APIRouter
from ..serializers import ok

router = APIRouter()

@router.get('/')
async def status():
    return ok('ok')
