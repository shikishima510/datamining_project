import logging

import uvicorn

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
