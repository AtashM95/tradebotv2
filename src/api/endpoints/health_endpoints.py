
from fastapi import APIRouter
from ..serializers import ok

router = APIRouter()

@router.get('/')
async def health():
    return ok('healthy')
