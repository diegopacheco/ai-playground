import unittest
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
sys.path.append('src/')
from src import predict

class TestStringMethods(unittest.TestCase):

    def test_portuguese_prediction_present(self):
        result = predict.predict("Silva")
        print(result)
        if not 'Portuguese' in result:
            self.fail("Should be a portuguese name")

if __name__ == '__main__':
    unittest.main()