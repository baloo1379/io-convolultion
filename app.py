from pypbm import PyPBM
import numpy as np
import argparse
import asyncio
import random
import copy
import time

# enable extra sleep for simulating long calculation
SLOW = False
# enable extra logging
LOG = True


class Filters:
    TEST = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    CHESS_FILTER = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    ONES = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    BLUR = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    OUTLINE = np.array([[-.25, -.5, -.25], [-.5, 3, -.5], [-.25, -.5, -.25]])
    OUTLINE2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    EMBOSS = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    BOTTOM_SOBEL = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    LEFT_SOBEL = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    RIGHT_SOBEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    TOP_SOBEL = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    @staticmethod
    def choose(name: str):
        name = name.upper()
        if name == 'BLUR':
            return Filters.BLUR
        elif name == 'SHARPEN':
            return Filters.SHARPEN
        elif name == 'OUTLINE':
            return Filters.OUTLINE
        elif name == 'OUTLINE2':
            return Filters.OUTLINE2
        elif name == 'EMBOSS':
            return Filters.EMBOSS
        elif name == 'BOTTOM_SOBEL':
            return Filters.BOTTOM_SOBEL
        elif name == 'LEFT_SOBEL':
            return Filters.LEFT_SOBEL
        elif name == 'RIGHT_SOBEL':
            return Filters.RIGHT_SOBEL
        elif name == 'TOP_SOBEL':
            return Filters.TOP_SOBEL
        elif name == 'ONES':
            return Filters.ONES
        elif name == 'CHESS_FILTER':
            return Filters.CHESS_FILTER
        else:
            raise ValueError("There is no such filter.")


class Convolution:

    @staticmethod
    async def produce_image(file, n, q):
        await asyncio.sleep(random.randint(1, 7) / 10)
        image = PyPBM(file, n)
        target = copy.deepcopy(image)
        target.max_value = 255
        if LOG:
            print("Image ready: ", image.info())
        await q.put(image)
        return target

    @staticmethod
    async def divide_pixels(images_q, corners_j, sides_j, rows_j, time_list) -> None:
        while True:
            image = await images_q.get()
            if LOG:
                print("Dividing image", image.id)

            for c in range(4):
                await corners_j.put((image, c))

            for s in range(4):
                await sides_j.put((image, s))

            for r in range(1, image.height - 1):
                await rows_j.put((image, r))

            time_list.append(time.perf_counter())

            print("All jobs scheduled.\nCalculating...")

            images_q.task_done()

    @staticmethod
    async def produce_corners(corners_j, corners_o):
        while True:
            if SLOW:
                await asyncio.sleep(random.randint(1, 5) / 10)
            image, side = await corners_j.get()
            if image.type != 'P3':
                arr_type = int
            else:
                arr_type = (int, 3)
            corner = np.zeros((3, 3), arr_type)
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
            if LOG:
                print(f"Extruded corner {side} from image {image.id}")
            await corners_o.put((image, position, corner))
            corners_j.task_done()

    @staticmethod
    async def produce_sides(sides_j, sides_o):
        while True:
            if SLOW:
                await asyncio.sleep(random.randint(1, 17) / 10)
            image, side = await sides_j.get()
            if image.type != 'P3':
                arr_type = int
            else:
                arr_type = (int, 3)
            if side == 0:  # top
                row = np.zeros((3, image.width), arr_type)
                for i in range(2):
                    row[i + 1] = image.pixels[i]
            elif side == 1:  # right
                row = np.zeros((image.height, 3), arr_type)
                for h in range(image.height):
                    row[h][0] = image.pixels[h][-2]
                    row[h][1] = image.pixels[h][-1]
                row = row.T  # transposition array from vertical to horizontal
            elif side == 2:  # bottom
                row = np.zeros((3, image.width), arr_type)
                for i in range(2):
                    row[i] = image.pixels[-2 + i]
            elif side == 3:  # left
                row = np.zeros((image.height, 3), arr_type)
                for h in range(image.height):
                    row[h][1] = image.pixels[h][-1]
                    row[h][2] = image.pixels[h][-2]
                row = row.T
            else:
                raise ValueError(f"wrong side", side)
            if LOG:
                print(f"Extruded side {side} from image {image.id}")
            await sides_o.put((image, side, row))
            sides_j.task_done()

    @staticmethod
    async def produce_rows(rows_j, rows_o):
        while True:
            if SLOW:
                await asyncio.sleep(random.randint(1, 17) / 10)
            image, r = await rows_j.get()
            if image.type != 'P3':
                arr_type = int
            else:
                arr_type = (int, 3)
            row = np.zeros((3, image.width), dtype=arr_type)
            row[0] = image.pixels[r - 1]
            row[1] = image.pixels[r]
            row[2] = image.pixels[r + 1]
            if LOG:
                print(f"Extruded row {r} from image {image.id}")
            await rows_o.put((image, r, row))
            rows_j.task_done()

    @staticmethod
    async def calculate_corner(corner_o, pixels_j, f):
        while True:
            image, coord, array = await corner_o.get()
            if LOG:
                print(f"Calculating corner {coord} for image {image.id}")
            value = await Convolution.calc(array, f)
            await pixels_j.put((image, coord, value))
            corner_o.task_done()

    @staticmethod
    async def calculate_side(pixels_i, pixels_o, f):
        while True:
            image, side, array = await pixels_i.get()
            if LOG:
                print(f"Calculating side {side} for image {image.id}")
            if len(array[0]) > 3:
                for i in range(len(array[0]) - 2):
                    value = await Convolution.calc(await Convolution.get_cell(array, 0, i), f)
                    x = 0
                    y = 0
                    if side == 0:
                        x = 0
                        y = i + 1
                    elif side == 1:
                        x = i + 1
                        y = image.width - 1
                    elif side == 2:
                        x = image.height - 1
                        y = i + 1
                    elif side == 3:
                        x = i + 1
                        y = 0
                    # print("s", (idx, (x, y), value))
                    await pixels_o.put((image, (x, y), value))
            pixels_i.task_done()

    @staticmethod
    async def calculate_row(pixels_i: asyncio.Queue, pixels_o: asyncio.Queue, worker_id: int, kernel: np.ndarray):
        while True:
            image, row, array = await pixels_i.get()
            if LOG:
                print(f"Calculating row {row} for image {image.id} by {worker_id}")
            for i in range(len(array[0]) - 2):
                value = await Convolution.calc(await Convolution.get_cell(array, 0, i), kernel)
                await pixels_o.put((image, (row, i + 1), value))
            pixels_i.task_done()

    @staticmethod
    async def calc(cell, kernel):
        cell_sum = 0
        rgb_sum = [0, 0, 0]
        cell_type = type(cell[0][0])
        for h in range(len(cell)):
            for w in range(len(cell[0])):
                if cell_type is not np.ndarray:
                    cell_sum += int(cell[h][w] * kernel[h][w])
                else:
                    rgb_sum[0] += int(cell[h][w][0] * kernel[h][w])
                    rgb_sum[1] += int(cell[h][w][1] * kernel[h][w])
                    rgb_sum[2] += int(cell[h][w][2] * kernel[h][w])
        if cell_type is not np.ndarray:
            cell_sum = cell_sum if cell_sum >= 0 else 0
            cell_sum = cell_sum if cell_sum < 255 else 255
        else:
            cell_sum = []
            for color in rgb_sum:
                color = color if color >= 0 else 0
                color = color if color < 255 else 255
                cell_sum.append(color)
        return cell_sum

    @staticmethod
    async def get_cell(array, x, y):
        if x >= len(array) - 2:
            raise IndexError(f"index {x} is out of bounds for axis 0 with size {len(array) - 2}")
        if y >= len(array[x]) - 2:
            raise IndexError(f"index {y} is out of bounds for axis {x} with size {len(array[x]) - 2}")
        return array[x:x + 3, y:y + 3]

    @staticmethod
    async def merge_pixels(pixels_o, target, lock):
        while True:
            image, coords, value = await pixels_o.get()
            if LOG:
                print(f"Merging pixels for image {image.id}. Current pixel {coords}")
            async with lock:
                target.pixels[coords] = value
            pixels_o.task_done()

    @staticmethod
    async def run(file: str, o: str, f: np.ndarray):
        start = time.perf_counter()
        time_list = [start]

        lock = asyncio.Lock()

        images_q = asyncio.Queue()
        corners_j = asyncio.Queue()
        corners_o = asyncio.Queue()
        sides_j = asyncio.Queue()
        sides_o = asyncio.Queue()
        rows_j = asyncio.Queue()
        rows_o = asyncio.Queue()
        pixels_j = asyncio.Queue()

        load = asyncio.create_task(Convolution.produce_image(file, 0, images_q))
        asyncio.create_task(Convolution.divide_pixels(images_q, corners_j, sides_j, rows_j, time_list))

        asyncio.create_task(Convolution.produce_corners(corners_j, corners_o))
        asyncio.create_task(Convolution.produce_sides(sides_j, sides_o))
        [asyncio.create_task(Convolution.produce_rows(rows_j, rows_o)) for i in range(1)]

        asyncio.create_task(Convolution.calculate_corner(corners_o, pixels_j, f))
        asyncio.create_task(Convolution.calculate_side(sides_o, pixels_j, f))
        [asyncio.create_task(Convolution.calculate_row(rows_o, pixels_j, i, f)) for i in range(1)]

        final_image = await load

        asyncio.create_task(Convolution.merge_pixels(pixels_j, final_image, lock))

        await images_q.join()
        await corners_j.join()
        await corners_o.join()
        await sides_j.join()
        await sides_o.join()
        await rows_j.join()
        await rows_o.join()
        await pixels_j.join()

        end_image = time_list[1] - time_list[0]
        end_calc = time.perf_counter() - time_list[1]
        print(f"image {final_image.id} loaded in {end_image:0.2f} seconds.")
        print(f"result {final_image.id} took {end_calc:0.2f} seconds.")
        final_image.save(o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make convolution filtering on image")
    parser.add_argument('input_file', metavar='I', type=str, help='name of the file to process')
    parser.add_argument('output_file', metavar='O', type=str, help='name of the file to the result')
    parser.add_argument('kernel', metavar='K', type=str, help='name of the filter to use', default='blur')
    args = parser.parse_args()
    try:
        i = args.input_file
        o = args.output_file
        k = Filters.choose(args.kernel)
    except ValueError as err:
        print(err)
    else:
        asyncio.run(Convolution.run(i, o, k))
