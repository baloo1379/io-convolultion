import asyncio
import fire
import random


class CLI:

    def __init__(self):
        print(f"Welcome in app.")
        asyncio.run(Image.convolution())


class Image:

    @staticmethod
    async def load_image():
        print("Getting image")
        await asyncio.sleep(random.randint(1, 3))
        print("IMAGE")
        return [[[0, 0, 0]], [0, 0, 0], [0, 0, 0]]

    @staticmethod
    async def mock_mask():
        await asyncio.sleep(random.randint(2, 5))
        print("MASK")
        return [[[1, 0, 1]], [0, 1, 0], [1, 0, 1]]

    @staticmethod
    async def convolution():
        image = asyncio.create_task(Image.load_image())
        image1 = asyncio.create_task(Image.load_image())
        image2 = asyncio.create_task(Image.load_image())

        await asyncio.gather(image, image1, image2)


if __name__ == "__main__":
    asyncio.run(Image.convolution())
