import unittest
from pathlib import Path

def check_file(self,f):
        my_file = Path(f)
        if not my_file.is_file():
            self.fail(f"{my_file} does not exist")

class TestStringMethods(unittest.TestCase):

    def test_data_is_present(self):
        check_file(self,"data/names/Portuguese.txt")
        check_file(self,"data/names/Arabic.txt")
        check_file(self,"data/names/Czech.txt")
        check_file(self,"data/names/Dutch.txt")
        check_file(self,"data/names/French.txt")
        check_file(self,"data/names/German.txt")
        check_file(self,"data/names/Greek.txt")
        check_file(self,"data/names/Irish.txt")
        check_file(self,"data/names/Italian.txt")
        check_file(self,"data/names/Japanese.txt")
        check_file(self,"data/names/Korean.txt")
        check_file(self,"data/names/Scottish.txt")
        check_file(self,"data/names/Spanish.txt")
        check_file(self,"data/names/Vietnamese.txt")

if __name__ == '__main__':
    unittest.main()