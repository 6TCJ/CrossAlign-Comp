
import requests
from io import BytesIO
from PIL import Image

image_id = "000000523244.jpg"

for protocol in ["https", "http"]:
    url = (
        f"{protocol}://images.cocodataset.org/"
        f"train2017/{image_id}"
    )

    print("\nTesting:", url)

    try:
        response = requests.get(
            url,
            timeout=(8, 20),
            headers={"User-Agent": "Mozilla/5.0"}
        )

        print("HTTP Status:", response.status_code)
        print("Content Type:", response.headers.get("Content-Type"))
        print("Bytes:", len(response.content))

        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image.verify()

        print("IMAGE VERIFIED")

    except Exception as e:
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR DETAIL:", str(e)[:500])
