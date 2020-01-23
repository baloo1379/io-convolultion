class PyPBM:

    def __init__(self, file_path):
        self.formats = ['P1']
        with open(file_path, encoding='UTF-8') as f:
            # first -> format type
            self.type = self._type(self._getline(f))
            # second -> sizes
            self.size = self._size(self._getline(f))

        f.close()

    @staticmethod
    def _getline(handle):
        curr_line = handle.readline().rstrip('\n')
        if not PyPBM._comment(curr_line):
            return curr_line
        else:
            return PyPBM._getline(handle)

    def _type(self, line):
        if line not in self.formats:
            raise TypeError(f"'{line}' Format not supported")
        else:
            return line

    @staticmethod
    def _size(line):
        sizes = line.split()
        if len(sizes) == 2:
            width, height = sizes
            width = int(width)
            height = int(height)
            return width, height
        else:
            raise ValueError(f"'{line}': Unsupported size format. Should be 'width height' ex. '12 5'.")

    @staticmethod
    def _comment(line):
        return line[0] == '#'


if __name__ == "__main__":
    p = PyPBM('p1.txt')
