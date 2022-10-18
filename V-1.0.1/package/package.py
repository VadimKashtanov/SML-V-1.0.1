######################### INSTS #########################

from package.insts.order import INSTS as INSTS_NAMES

INSTS_NAMES = [
	name.upper() for name in INSTS_NAMES
]

INSTS = []
INSTS_DICT = {}

for _id, inst_name in enumerate(INSTS_NAMES):
	exec(f"from package.insts.{inst_name.lower()}.py.{inst_name.lower()} import {inst_name}")
	exec(f"INSTS += [{inst_name}]")
	exec(f"INSTS_DICT['{inst_name}'] = {inst_name}")
	exec(f"{inst_name}._id = {_id}")
	exec(f"{inst_name}.ID = {_id}")

########################  SCORES ########################

from package.scores.order import SCORES

SCORES_NAMES = [
	name.upper() for name in SCORES
]

SCORES = []

for score in SCORES_NAMES:
	exec(f"from package.scores.{score.lower()}.py.{score.lower()} import {score}")
	exec(f"SCORES += [{score}]")

######################### OPTIS #########################

from package.optis.order import OPTIS

OPTIS_NAMES = [
	name.upper() for name in OPTIS
]

OPTIS = []

for opti in OPTIS_NAMES:
	exec(f"from package.optis.{opti.lower()}.py.{opti.lower()} import {opti}")
	exec(f"OPTIS += [{opti}]")

###################################################################################
###################################################################################
###################################################################################

####################### TEST_MODELS #####################

TEST_MODELS_NAMES = [
	("TEST_MDL_" + name) for name in INSTS_NAMES
]

TEST_MODELS = []

for inst_name in INSTS_NAMES:
	exec(f"from package.insts.{inst_name.lower()}.py.test_mdl_{inst_name.lower()} import TEST_MDL_{inst_name.upper()}")
	exec(f"TEST_MODELS += [TEST_MDL_{inst_name.upper()}]")

####################### TEST_SCORES #####################

TEST_SCORES_NAMES = [
	("TEST_SCORE_" + name) for name in SCORES_NAMES
]

TEST_SCORES = []

for score in SCORES_NAMES:
	exec(f"from package.scores.{score.lower()}.py.test_score_{score.lower()} import TEST_SCORE_{score.upper()}")
	exec(f"TEST_SCORES += [TEST_SCORE_{score.upper()}]")

####################### TEST_OPTIS #####################

TEST_OPTIS_NAMES = [
	("TEST_OPTI_" + name) for name in OPTIS_NAMES
]

TEST_OPTIS = []

for opti in OPTIS_NAMES:
	exec(f"from package.optis.{opti.lower()}.py.test_opti_{opti.lower()} import TEST_OPTI_{opti.upper()}")
	exec(f"TEST_OPTIS += [TEST_OPTI_{opti.upper()}]")