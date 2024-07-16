import pydantic


class StreamResponse(pydantic.BaseModel):
    message: str
    status: bool
    successfully_uploaded: bool
