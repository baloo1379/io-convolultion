import tokenize as t
import numpy as np


class PyPBM:
    formats = ['P1', 'P2', 'P3']
    white = [0, 4, 60]
    NAME = 1
    NUMBER = 2
    COMMENT = 60

    def __init__(self, file_path):
        with open(file_path, encoding='UTF-8') as f:
            tokens = t.generate_tokens(f.read)

            # Format
            token = self.get_next_token(tokens)
            if token.type == self.NAME and token.string in self.formats:
                self.type = token.string
            else:
                raise ValueError("wrong header")

            # Width
            token = self.get_next_token(tokens)
            if token.type == self.NUMBER:
                self.width = int(token.string)
            else:
                raise ValueError("wrong width syntax")

            # Height
            token = self.get_next_token(tokens)
            if token.type == self.NUMBER:
                self.height = int(token.string)
            else:
                raise ValueError("wrong height syntax")

            # Max value of pixel (if present)
            if self.type != 'P1':
                token = self.get_next_token(tokens)
                if token.type == self.NUMBER:
                    self.max_value = int(token.string)
                else:
                    raise ValueError("wrong height syntax")
            else:
                self.max_value = 1

            # Pixels array
            self.pixels = np.zeros((self.height, self.width), int)

            for i in range(self.height):
                row = np.empty(self.width)
                for j in range(self.width):
                    if self.type != 'P3':
                        token = self.get_next_token(tokens)
                        if token.type != self.NUMBER:
                            raise ValueError("wrong pixel format at ", token.start)
                        row[j] = int(token.string)
                    else:
                        token = self.get_next_token(tokens)
                        if token.type != self.NUMBER:
                            raise ValueError("wrong pixel format at ", token.start)
                        r = int(token.string)
                        token = self.get_next_token(tokens)
                        if token.type != self.NUMBER:
                            raise ValueError("wrong pixel format at ", token.start)
                        g = int(token.string)
                        token = self.get_next_token(tokens)
                        if token.type != self.NUMBER:
                            raise ValueError("wrong pixel format at ", token.start)
                        b = int(token.string)
                        row[j] = self.rgb_to_int(r, g, b, self.max_value)
                self.pixels[i] = row
        f.close()

    def __repr__(self):
        res = f"Format: {self.type}\nSize: {self.width} x {self.height}\n"
        res += f"Max value: {self.max_value}\n" if self.type == 'P2' or self.type == 'P3' else ""
        res += f"{self.pixels}"
        return res

    @staticmethod
    def get_next_token(gen):
        token = next(gen)
        if token.type not in PyPBM.white:
            return token
        else:
            return PyPBM.get_next_token(gen)

    @staticmethod
    def scale_number(number, current_max):
        return int(number / current_max) * 255

    @staticmethod
    def rgb_to_int(r, g, b, current_max):
        rgb = PyPBM.scale_number(r, current_max)
        rgb = (rgb << 8) + PyPBM.scale_number(g, current_max)
        rgb = (rgb << 8) + PyPBM.scale_number(b, current_max)
        return rgb

    @staticmethod
    def int_to_rgb(rgb, prev_max):
        red = (rgb >> 16) & 0xFF
        green = (rgb >> 8) & 0xFF
        blue = rgb & 0xFF
        return PyPBM.scale_number(red, prev_max), PyPBM.scale_number(green, prev_max), PyPBM.scale_number(blue, prev_max)


if __name__ == "__main__":
    p = PyPBM('p3_multiline_bw.ppm')
    print(p)

