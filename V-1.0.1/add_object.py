def build():
	pass

if __name__ == "__main__":

	from os import argv, listdir, system


	#	Checking object
	_object = argv[1]
	assert _object in ('inst', 'opti', 'score')

	#	Checking name
	_name = argv[2]
	assert not _name in listdir(f'package/{_object}')

	#	Build modes
	_modes = argv[3:]
	assert all(not mod in _modes for mod in _modes)	#verifier qu'ils ne sont pas les memes

	#	Confirming
	if input(f"You want to build package/{_object}/{_name} with mods {_modes} ? (yes / no)").replace(' ','') == 'yes':
		######################## General ####################################
		### 	Building Object directory 	###
		system(f"mkdir package/{_object}/{_name}")

		system(f"mkdir package/{_object}/{_name}/head")
		system(f"mkdir package/{_object}/{_name}/py")
		system(f"mkdir package/{_object}/{_name}/src")

		###		Files that are required for python
		system(f"touch package/{_object}/{_name}/__init__.py")
		build_python(f"package/{_object}/{_name}/py/{_name}.py", _object, _name)
		build_python_test_package(f"package/{_object}/{_name}/py/test_{_object}_{name}.py", _object, _name)

		###		General .cuh header
		build_head(f"package/{_object}/{_name}/head/{_name}.cuh.py", _object, _name)

		###################### C/Cuda Computing Modes #########################
		###	Building all Mods ###
		for mod in _modes:
			build_head_mod(f"package/{_object}/{_name}/head/{_name}_{mod}.cuh.py", _object, _name)
			system(f"mkdir package/{_object}/{_name}/src{mod}")

	else:
		print("Cancelling")