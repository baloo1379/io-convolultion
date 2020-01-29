import asyncio
import fire
import random
import copy
import time
from pypbm import PyPBM, np


class Filters:
    CHESS_FILTER = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    ONES = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    BLUR = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    OUTLINE = np.array([[-.25, -.5, -.25], [-.5, 3, -.5], [-.25, -.5, -.25]])
    EMBOSS = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    DARKER = np.array([[0, 0, 0], [0, 0.5, 0], [0, 0, 0]])


class CLI:

    def __init__(self, i: str, o: str, k: str):
        print(f"Welcome in app.")
        f = Filters.ONES
        if k.lower() == "blur":
            f = Filters.BLUR
        elif k.lower() == "sharpen":
            f = Filters.SHARPEN
        elif k.lower() == "outline":
            f = Filters.OUTLINE
        elif k.lower() == "emboss":
            f = Filters.EMBOSS
        elif k.lower() == "chess":
            f = Filters.CHESS_FILTER
        asyncio.run(Convolution.run(i, o, f))


class Convolution:

    @staticmethod
    async def produce_image(file, n, q):
        await asyncio.sleep(random.randint(1, 7) / 10)
        image = PyPBM(file, n)
        target = copy.deepcopy(image)
        target.max_value = 255
        print("Image ready: ", image.info())
        await q.put(image)
        return target

    @staticmethod
    async def divide_pixels(images_q, corners_j, sides_j, rows_j, time_list) -> None:
        while True:
            image = await images_q.get()
            print("Dividing image", image.id)

            for c in range(4):
                await corners_j.put((image, c))

            for s in range(4):
                await sides_j.put((image, s))

            for r in range(1, image.height - 1):
                await rows_j.put((image, r))

            time_list.append(time.perf_counter())

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
            # print(f"Extruded corner {side} from image {image.id}")
            await corners_o.put((image, position, corner))
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

            # print(f"Extruded side {side} from image {target.id}")
            await sides_o.put((image, side, row))
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
            # print(f"Extruded row {r} from image {target.id}")
            await rows_o.put((image, r, row))
            rows_j.task_done()

    @staticmethod
    async def calculate_corner(corner_o, pixels_j, f):
        while True:
            target, coord, array = await corner_o.get()
            # print("Calculating for image", target.id, "on corner", coord)
            value = await Convolution.calc(array, f)
            await pixels_j.put((target, coord, value))
            corner_o.task_done()

    @staticmethod
    async def calculate_side(pixels_i, pixels_o, f):
        while True:
            target, side, array = await pixels_i.get()
            # print("Calculating for image", target.id, "on side", side)
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
                        y = target.width - 1
                    elif side == 2:
                        x = target.height - 1
                        y = i + 1
                    elif side == 3:
                        x = i + 1
                        y = 0
                    # print("s", (idx, (x, y), value))
                    await pixels_o.put((target, (x, y), value))
            pixels_i.task_done()

    @staticmethod
    async def calculate_row(pixels_i, pixels_o, worker_id, f):
        while True:
            # await asyncio.sleep(0.1)
            target, row, array = await pixels_i.get()
            # print("Calculating for image", target.id, "on row", row, "by", worker_id)
            for i in range(len(array[0]) - 2):
                value = await Convolution.calc(await Convolution.get_cell(array, 0, i), f)
                # print("r", (idx, (row, i + 1), value))
                await pixels_o.put((target, (row, i + 1), value))
            pixels_i.task_done()

    @staticmethod
    async def calc(cell, kernel):
        cell_sum = 0
        filter_sum = np.sum(kernel)
        for h in range(len(cell)):
            for w in range(len(cell[0])):
                cell_sum += int(cell[h][w] * kernel[h][w])
        return cell_sum

    @staticmethod
    async def get_cell(array, x, y):
        if x >= len(array) - 2:
            raise IndexError(f"index {x} is out of bounds for axis 0 with size {len(array) - 2}")
        if y >= len(array[x]) - 2:
            raise IndexError(f"index {y} is out of bounds for axis {x} with size {len(array[x]) - 2}")
        return array[x:x + 3, y:y + 3]

    @staticmethod
    async def merge_pixels(pixels_o, final_image, lock):
        while True:
            target, coords, value = await pixels_o.get()
            # print(idx, coords, value)
            async with lock:
                value = value if value >= 0 else 0
                if value > final_image.max_value:
                    value = final_image.max_value
                final_image.pixels[coords] = value
            pixels_o.task_done()

    @staticmethod
    async def run(file: str, o: str, f: np.ndarray):
        lock = asyncio.Lock()

        start = time.perf_counter()
        time_list = [start]
        images_q = asyncio.Queue()
        # jobs
        corners_j = asyncio.Queue()
        corners_o = asyncio.Queue()
        sides_j = asyncio.Queue()
        sides_o = asyncio.Queue()
        rows_j = asyncio.Queue()
        rows_o = asyncio.Queue()
        pixels_j = asyncio.Queue()

        load = asyncio.create_task(Convolution.produce_image(file, 0, images_q))
        divide_task = asyncio.create_task(Convolution.divide_pixels(images_q, corners_j, sides_j, rows_j, time_list))

        corners_task = asyncio.create_task(Convolution.produce_corners(corners_j, corners_o))
        sides_task = asyncio.create_task(Convolution.produce_sides(sides_j, sides_o))
        rows_task = [asyncio.create_task(Convolution.produce_rows(rows_j, rows_o)) for i in range(8)]

        calculation_c = asyncio.create_task(Convolution.calculate_corner(corners_o, pixels_j, f))
        calculation_s = asyncio.create_task(Convolution.calculate_side(sides_o, pixels_j, f))
        calculation_r = [asyncio.create_task(Convolution.calculate_row(rows_o, pixels_j, i, f)) for i in range(8)]

        final_image = await load

        print_t = asyncio.create_task(Convolution.merge_pixels(pixels_j, final_image, lock))

        # await asyncio.gather(*calculation_r)

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
        # final_image.cut_edges()
        final_image.update_max_value()
        print(f"image {final_image.id} loaded in {end_image:0.2f} seconds.")
        print(f"result {final_image.id} took {end_calc:0.2f} seconds.")
        final_image.save(o)


if __name__ == "__main__":
    pass
    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # b = np.array([[0, 0, 0], [0, 0.5, 0], [0, 0, 0]])
    # print(Convolution.calc(a, Filters.OUTLINE))
    asyncio.run(Convolution.run("balloons.ascii.pgm", "b.pgm", Filters.OUTLINE))
    # fire.Fire(CLI)
