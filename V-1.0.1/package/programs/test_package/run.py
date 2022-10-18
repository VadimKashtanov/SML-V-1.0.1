from package.package import TEST_MODELS, TEST_SCORES, TEST_OPTIS
from kernel.py.test_package import test_package

if __name__ == "__main__":
	bins = test_package(TEST_MODELS, TEST_SCORES, TEST_OPTIS)

	with open("save.bin", 'wb') as co:
		co.write(bins)