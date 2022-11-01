# What to Do #

Add all programs to the ```Makefile```

Build the programs of the package with all ```order.py``` in each ```/insts``` ```/optis``` ```/scores```

```
make
```

## First Steps ##

If there is ```test_package``` and ```test_package_python``` programs, run them.

### This package ###

If you wan to test the package with some data, build ```/my_test``` in ```/tests```.

Add ```/bin```, ```/config_files```, ```/python```, ```my_data``` and past in the last dir your data.

Save in ```/bins``` all binary files like data in ```Data_t``` format, models in ```Mdl_t``` format ...

In ```/config_files``` same file with configuration to Optimize or build models.

### Building Models ###

You can build some feed-forward networks (stack of layers) using 

```

./cli_stack_model <path to the config file>

```

or juste making in ```/python``` file with ```class <My_Model>(Heritance):``` and build them with ```.bins()```

### Build Data ###

You can build in ```Data_t``` format all data using python file, or some programs to do it.

### Optimize ###

Write python file in ```/config_files``` a dictionnay with all required parameters (```optimize_mdl/compile.py```)

```

./optimize_mdl <path to the config file>

```

### Test you model ###

You can make a ```.py``` file but also use ```./test_mdl```

```

./test_mdl optimized_mdl.bin test_data.bin <batch> <line>

```
