import tokenize as t
import numpy as np


class PyPBM:
    formats = ['P1', 'P2', 'P3']
    white = [0, 4, 60]
    NAME = 1
    NUMBER = 2
    COMMENT = 60

    def __init__(self, path, n):
        self.file_name = path
        self.id = n
        self.type = None
        self.width = 0
        self.height = 0
        self.max_value = 0
        self.pixels = np.zeros((self.height, self.width), int)
        with open(path, encoding='UTF-8') as f:
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
            if self.type != 'P3':
                self.pixels = np.zeros((self.height, self.width), dtype=int)
            else:
                self.pixels = np.zeros((self.height, self.width), dtype=(int, 3))

            for i in range(self.height):
                row = self.pixels[i]
                for j in range(self.width):
                    if self.type != 'P3':
                        token = self.get_next_token(tokens)
                        if token.type != self.NUMBER:
                            raise ValueError("wrong pixel format at ", token.start)
                        a = int(token.string)
                        row[j] = a
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
                        row[j] = (r, g, b)
                self.pixels[i] = row
        f.close()

    def __repr__(self):
        res = f"Image: {self.file_name} id:{self.id}"
        res += f"Format: {self.type}\nSize: {self.width} x {self.height}\n"
        res += f"Max value: {self.max_value}\n" if self.type != 'P1' else ""
        res += f"{self.pixels}"
        return res

    def update_max_value(self):
        self.max_value = np.amax(self.pixels)

    def info(self):
        return f"Image: {self.file_name} id: {self.id}"

    def save(self, new_file):
        with open(new_file, 'w') as f:
            f.write(self.type+"\n")
            f.write(str(self.width)+" ")
            f.write(str(self.height)+"\n")
            if self.type != 'P1':
                f.write(str(self.max_value)+"\n")
            for h in range(self.height):
                row = ""
                for w in range(self.width):
                    if self.type != 'P3':
                        row += str(self.pixels[h][w]) + " "
                    else:
                        for color in self.pixels[h][w]:
                            row += str(color) + " "
                        # row += " "
                f.write(row+"\n")

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


if __name__ == "__main__":
    p = PyPBM('pbmlib.ascii.ppm', 0)
    p.save("output/p.ppm")

