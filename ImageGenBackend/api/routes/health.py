from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(session: bool = True):
    """Health check endpoint"""
    return session


@router.get("/favicon.ico")
def favicon():
    """Favicon endpoint to handle browser requests"""
    # Return a simple 204 No Content response
    from fastapi.responses import Response
    return Response(status_code=204)
