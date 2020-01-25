import asyncio
import fire
import random
from pypbm import PyPBM, np


class CLI:

    def __init__(self):
        print(f"Welcome in app.")
        asyncio.run(Convolution.run())


class Convolution:

    @staticmethod
    async def load_image():
        print("Getting image")
        image = PyPBM('p1.pbm')
        print(f"Image: {image.file_name}")
        return image

    @staticmethod
    async def divide_pixels(image: PyPBM):
        pass
        # TODO rozdzielić na krawędzie, narożniki i środek
        lu = await Convolution.make_corner(image.pixels[0:2], side=0)
        ru = await Convolution.make_corner(image.pixels[0:2], side=1)
        lb = await Convolution.make_corner(image.pixels[-2:], side=2)
        rb = await Convolution.make_corner(image.pixels[-2:], side=3)
        return lu, ru, rb, lb


    '''
        0 - left upper corner
        1 - right upper corner
        2 - left bottom corner
        3 - right bottom corner
    '''
    @staticmethod
    async def make_corner(pixel_array, side=0):
        corner = np.zeros((3, 3), int)
        if side == 0:
            corner[1][1] = pixel_array[0][0]
            corner[1][2] = pixel_array[0][1]
            corner[2][1] = pixel_array[1][0]
            corner[2][2] = pixel_array[1][1]
        elif side == 1:
            corner[1][0] = pixel_array[0][-2]
            corner[1][1] = pixel_array[0][-1]
            corner[2][0] = pixel_array[1][-2]
            corner[2][1] = pixel_array[1][-1]
        elif side == 2:
            corner[0][0] = pixel_array[0][-2]
            corner[0][1] = pixel_array[0][-1]
            corner[1][0] = pixel_array[1][-2]
            corner[1][1] = pixel_array[1][-1]
        elif side == 3:
            corner[0][1] = pixel_array[0][0]
            corner[0][2] = pixel_array[0][1]
            corner[1][1] = pixel_array[1][0]
            corner[1][2] = pixel_array[1][1]
        else:
            raise ValueError("wrong side")
        return corner

    @staticmethod
    async def prepare_tasks():
        image = await Convolution.load_image()
        return await Convolution.divide_pixels(image)

    @staticmethod
    async def run():
        load = asyncio.create_task(Convolution.prepare_tasks())
        data = await load
        print(data)


if __name__ == "__main__":
    asyncio.run(Convolution.run())
