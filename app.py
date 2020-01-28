import asyncio
import fire
import random
from pypbm import PyPBM, np

filt = np.ones((3, 3), int)

'''
How the pixel should look between threads
- It should be a tuple
- [0] - id of image
- [1] - (x, y)
- [2] - array of pixel and element around it
'''


class CLI:

    def __init__(self):
        print(f"Welcome in app.")
        asyncio.run(Convolution.run())


class Convolution:

    @staticmethod
    async def produce_image(n, q):
        await asyncio.sleep(random.randint(1, 7) / 10)
        image = PyPBM('p2.pgm', n)
        print("Image ready: ", image.info())
        await q.put(image)
        return np.zeros((image.height, image.width), int)

    @staticmethod
    async def divide_pixels(images_q, corners_j, sides_j, rows_j) -> None:
        while True:
            image = await images_q.get()
            print("Dividing image", image.id)

            for c in range(4):
                await corners_j.put((image, c))

            for s in range(4):
                await sides_j.put((image, s))

            for r in range(1, image.height - 1):
                await rows_j.put((image, r))

            print("All jobs scheduled")

            images_q.task_done()

    @staticmethod
    async def produce_corners(corners_j, corners_o):
        while True:
            image, side = await corners_j.get()
            # await asyncio.sleep(random.randint(1, 17) / 10)
            corner = np.zeros((3, 3), int)
            position = (0, 0)
            if side == 0:  # left upper
                pixel_array = image.pixels[0:2]
                corner[1][1] = pixel_array[0][0]
                corner[1][2] = pixel_array[0][1]
                corner[2][1] = pixel_array[1][0]
                corner[2][2] = pixel_array[1][1]
            elif side == 1:  # right upper
                pixel_array = image.pixels[0:2]
                corner[1][0] = pixel_array[0][-2]
                corner[1][1] = pixel_array[0][-1]
                corner[2][0] = pixel_array[1][-2]
                corner[2][1] = pixel_array[1][-1]
                position = (0, image.width - 1)
            elif side == 2:  # right bottom
                pixel_array = image.pixels[-2:]
                corner[0][0] = pixel_array[0][-2]
                corner[0][1] = pixel_array[0][-1]
                corner[1][0] = pixel_array[1][-2]
                corner[1][1] = pixel_array[1][-1]
                position = (image.height - 1, 0)
            elif side == 3:  # left bottom
                pixel_array = image.pixels[-2:]
                corner[0][1] = pixel_array[0][0]
                corner[0][2] = pixel_array[0][1]
                corner[1][1] = pixel_array[1][0]
                corner[1][2] = pixel_array[1][1]
                position = (image.height - 1, image.width - 1)
            else:
                raise ValueError(f"wrong side", side)
            print(f"Extruded corner {side} from image {image.id}")
            await corners_o.put((image.id, position, corner, image.height, image.width))
            corners_j.task_done()

    @staticmethod
    async def produce_sides(sides_j, sides_o):
        while True:
            image, side = await sides_j.get()
            # await asyncio.sleep(random.randint(1, 17) / 10)
            if side == 0:  # top
                row = np.zeros((3, image.width), int)
                for i in range(2):
                    row[i + 1] = image.pixels[i]
            elif side == 1:  # right
                row = np.zeros((image.height, 3), int)
                for h in range(image.height):
                    row[h][0] = image.pixels[h][-2]
                    row[h][1] = image.pixels[h][-1]
                row = row.T  # transposition array from vertical to horizontal
            elif side == 2:  # bottom
                row = np.zeros((3, image.width), int)
                for i in range(2):
                    row[i] = image.pixels[-2 + i]
            elif side == 3:  # left
                row = np.zeros((image.height, 3), int)
                for h in range(image.height):
                    row[h][1] = image.pixels[h][-1]
                    row[h][2] = image.pixels[h][-2]
                row = row.T
            else:
                raise ValueError(f"wrong side", side)

            print(f"Extruded side {side} from image {image.id}")
            await sides_o.put((image.id, side, row, image.height, image.width))
            sides_j.task_done()

    @staticmethod
    async def produce_rows(rows_j, rows_o):
        while True:
            image, r = await rows_j.get()
            # await asyncio.sleep(random.randint(1, 17) / 10)
            row = np.zeros((3, image.width), int)
            row[0] = image.pixels[r - 1]
            row[1] = image.pixels[r]
            row[2] = image.pixels[r + 1]
            print(f"Extruded row {r} from image {image.id}")
            await rows_o.put((image.id, r, row, image.height, image.width))
            rows_j.task_done()

    @staticmethod
    async def calculate_corner(corner_o, pixels_j):
        while True:
            idx, coord, array, h, w = await corner_o.get()
            print("Calculating for image", idx, "on corner", coord)
            value = await Convolution.calc(array)
            await pixels_j.put((idx, coord, value))
            corner_o.task_done()

    @staticmethod
    async def calculate_side(pixels_i, pixels_o):
        while True:
            idx, side, array, h, w = await pixels_i.get()
            print("Calculating for image", idx, "on side", side)
            if len(array[0]) > 3:
                for i in range(len(array[0]) - 2):
                    value = await Convolution.calc(await Convolution.get_cell(array, 0, i))
                    x = 0
                    y = 0
                    if side == 0:
                        x = 0
                        y = i + 1
                    elif side == 1:
                        x = i + 1
                        y = w-1
                    elif side == 2:
                        x = h-1
                        y = i + 1
                    elif side == 3:
                        x = i + 1
                        y = 0
                    # print("s", (idx, (x, y), value))
                    await pixels_o.put((idx, (x, y), value))
            pixels_i.task_done()

    @staticmethod
    async def calculate_row(pixels_i, pixels_o, worker_id):
        while True:
            idx, row, array, h, w = await pixels_i.get()
            print("Calculating for image", idx, "on row", row, "by", worker_id)
            if len(array[0]) > 3:
                for i in range(len(array[0]) - 2):
                    value = await Convolution.calc(await Convolution.get_cell(array, 0, i))
                    # print("r", (idx, (row, i + 1), value))
                    await pixels_o.put((idx, (row, i + 1), value))
            pixels_i.task_done()

    @staticmethod
    async def calc(cell):
        cell_sum = 0
        for h in range(len(cell)):
            for w in range(len(cell[0])):
                cell_sum += cell[h][w] * filt[h][w]
        return cell_sum

    @staticmethod
    async def get_cell(array, x, y):
        if x >= len(array) - 2:
            raise IndexError(f"index {x} is out of bounds for axis 0 with size {len(array) - 2}")
        if y >= len(array[x]) - 2:
            raise IndexError(f"index {y} is out of bounds for axis {x} with size {len(array[x]) - 2}")
        return array[x:x + 3, y:y + 3]

    @staticmethod
    async def save(pixels_o, final_image, lock):
        while True:
            idx, coords, value = await pixels_o.get()
            # print(idx, coords, value)
            async with lock:
                final_image[coords] = value
            pixels_o.task_done()

    @staticmethod
    async def run():
        lock = asyncio.Lock()

        images_q = asyncio.Queue()
        # jobs
        corners_j = asyncio.Queue()
        corners_o = asyncio.Queue()
        sides_j = asyncio.Queue()
        sides_o = asyncio.Queue()
        rows_j = asyncio.Queue()
        rows_o = asyncio.Queue()
        pixels_j = asyncio.Queue()

        load = asyncio.create_task(Convolution.produce_image(0, images_q))
        divide_task = asyncio.create_task(Convolution.divide_pixels(images_q, corners_j, sides_j, rows_j))

        corners_task = asyncio.create_task(Convolution.produce_corners(corners_j, corners_o))
        sides_task = asyncio.create_task(Convolution.produce_sides(sides_j, sides_o))
        rows_task = asyncio.create_task(Convolution.produce_rows(rows_j, rows_o))

        calculation_c = asyncio.create_task(Convolution.calculate_corner(corners_o, pixels_j))
        calculation_s = asyncio.create_task(Convolution.calculate_side(sides_o, pixels_j))
        calculation_r = [asyncio.create_task(Convolution.calculate_row(rows_o, pixels_j, i)) for i in range(1)]

        final_image = await load

        print_t = asyncio.create_task(Convolution.save(pixels_j, final_image, lock))

        # await asyncio.gather(*load)

        await images_q.join()
        await corners_j.join()
        await corners_o.join()
        await sides_j.join()
        await sides_o.join()
        await rows_j.join()
        await rows_o.join()
        await pixels_j.join()

        print(final_image)


if __name__ == "__main__":
    asyncio.run(Convolution.run())
