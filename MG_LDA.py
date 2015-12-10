



if __name__ == '__main__':
	import DataModule
	dataHandler = DataModule.DataHandler()


	X = dataHandler.get_data_sentenced()

	print X