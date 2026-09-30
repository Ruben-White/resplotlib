import os
import uuid

import contextily as cx
from dotenv import load_dotenv

load_dotenv()


def register_basemaps():
    # Avoid being blocked by tile providers by using a unique user agent
    cx.tile.USER_AGENT = f"resplotlib-{uuid.uuid4().hex}"

    # Patch the CartoDB URLs with the API key from the environment variable
    for key in cx.providers["CartoDB"]:
        cx.providers.CartoDB[key] = cx.providers.CartoDB[key](
            url=f"https://basemaps.cartocdn.com/rastertiles/{{variant}}/{{z}}/{{x}}/{{y}}.png?key={os.environ.get('RESPLOTLIB_CARTODB_API_KEY')}"
        )


__basemaps__ = register_basemaps()
