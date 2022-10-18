#! /usr/bin/python3

'''
Ici le but est de juste écrire les fichiers en .cu qui vont contenire les arrays ordonnés d'objets.
Par exemple pour les instructions, il faut une liste des fonctions de `forward` i.e train_f `INST_FORWARD[INSTS]`
Dans ce cas il faudra juste ecrire le nom de la liste puis, une ligne par une ligne `inst0_forward`, `inst1_forward` ...

Ce script python code la création et l'écriture des fichiers de liste pour les instructions, les scores, les optimisateurs et les genetics.

La structure de tout paquet est la suivante:

/package
	/insts
		/inst0
		/inst1
		ordre.py
		insts.cuh
		insts.cu
	/optis
		/opti0
		/opti1
		ordre.py
		optis.cuh
		optis.cu
	/scores
		/score0
		/score1
		ordre.py
		scores.cuh
		scores.cu
	/gtics
		/gtic0
		/gtic1
		ordre.py
		gtics.cuh
		gtics.cu

	/programs
		/program0
			main.cu
		/program1
			main.cu

On va cree les insts.cuh, optis.cuh, scores.cuh, gtics.cuh, insts.cu, optis.cu, scores.cu, gtics.cu, meta.cuh

Pour ça on va utiliser les classes de chaque objets qui sont implémenté dans les fichiers de chaque objet

On pourrait utiliser os.listdir() mais comment choisire l'ordre ? Par Ordre alphabetique ?
Pour plus de flexibilité on utilise un fichier `ordre.py` qui se trouvera dans chaque dossier d'objet, qui contiendra 
`ordre = ["nom0", "nom1" ...]`


PS: J'avais pour idée avant (meme depuis plusieurs mois) de cree a partire de cette architecture (qui aurait été une sorte de package_pre_construit)
un autre dossier de la forme de mes anciennes architectures (dont la forme c'est bien ancré depuis 1ans/1.5 ans).
Je considerais que mon architecture était plus logique. Elle est bonne, et bien construite, elle est meme exellente pour mes recherches
d'optimisation et de simplicité. Mais j'ai maitrisé la chose et je migre vers quelque chose ou seulement le kernel est comme ça.
Je pensais avant de faire une sorte de generateur de lib qui cree a partire d'un package_pre_build et du kernel les programs qu'il contient
en creant une autre lib temporaire.

En réalité ça fait 2 ans que mon projet a la meme forme plus ou moins (meme sans le cuda), et donc le 11 octobre 2022 je change vers la version finale.

'''


#################################################################################################################
########################################## Insts ################################################################
#################################################################################################################

from package.insts.order import INSTS

insts_cuh = "#pragma once\n"
insts_cu = '#include "package/insts/insts.cuh"\n\n'

for inst in INSTS:
	insts_cuh += f'#include "package/insts/{inst}/head/{inst}.cuh"\n'

for inst in INSTS:
	exec(f"from package.insts.{inst}.py.{inst} import {inst.upper()}")

insts_cu += "uint inst_params[INSTS] = {\n"
for inst in INSTS:
	args_len = len(eval(f"{inst.upper()}.params_names"))
	insts_cu += f"\t{args_len}, //{inst} \n"
insts_cu += "};\n\n"

insts_cu += "const char* inst_name[INSTS] = {\n"
for inst in INSTS:
	insts_cu += '\t"' + inst + '",\n'
insts_cu += "};\n\n"

for inst in INSTS:
	args = eval(f"{inst.upper()}.params_names")
	insts_cu += f"static const char* {inst}_params_names[{len(args)}] = "+ "{\n"
	for arg in args:
		insts_cu += '\t"' + arg + '",\n'
	insts_cu += '};\n\n'

arrays = {
	"const char** inst_param_name[INSTS] = {" : "_params_names",
	"check_f INST_CHECK[INSTS] = {" : "_check",
	"cpu_f INST_CPU[INSTS] = {" : "_cpu",
	"use_f INST_USE[INSTS] = {" : "_use",
	"train_f INST_FORWARD[INSTS] = {" : "_forward",
	"train_f INST_BACKWARD[INSTS] = {" : "_backward"
}

for key,value in arrays.items():
	insts_cu += key + '\n'
	for inst in INSTS:
		insts_cu += '\t' + inst + value + ',\n'
	insts_cu += "};\n\n"

with open("package/insts/insts.cu", "w") as co:
	co.write(insts_cu)

with open("package/insts/insts.cuh", "w") as co:
	co.write(insts_cuh)

########################################################################################################

#################################################################################################################
########################################## Optis ################################################################
#################################################################################################################

from package.optis.order import OPTIS

optis_cuh = "#pragma once\n"
optis_cu = '#include "package/optis/optis.cuh"\n\n'

for opti in OPTIS:
	optis_cuh += f'#include "package/optis/{opti}/head/{opti}.cuh"\n'

for opti in OPTIS:
	exec(f"from package.optis.{opti}.py.{opti} import {opti.upper()}")

arrays = {
	"uint OPTI_MIN_ECHOPES[OPTIS] = {" : "_min_echopes",
	"void* (*OPTI_OPTI_SPACE_MK_ARRAY[OPTIS])(Opti_t * opti) = {" : "_space_mk",
	"void (*OPTI_OPTI_SET_ONE_ARG_ARRAY[OPTIS])(Opti_t * opti, char * name, char * value) = {" : "_set_one_arg",
	"void (*OPTI_OPTIMIZE_ARRAY[OPTIS])(Opti_t * opti) = {" : "_optimize",
	"void (*OPTI_FREE_OPTI_ARRAY[OPTIS])(Opti_t * opti) = {" : "_optimize",
	"const uint OPTI_CONST_AMOUNT[OPTIS] = {" : "_CONSTS_AMOUNT",
	"const char ** OPTI_CONST_ARRAY[OPTIS] = {" : "_CONST_ARRAY"
}

for key,value in arrays.items():
	optis_cu += key + '\n'
	for opti in OPTIS:
		optis_cu += '\t' + opti.upper() + value + ',\n'
	optis_cu += "};\n\n"

with open("package/optis/optis.cu", "w") as co:
	co.write(optis_cu)

with open("package/optis/optis.cuh", "w") as co:
	co.write(optis_cuh)

########################################################################################################

#################################################################################################################
########################################## Scores ################################################################
#################################################################################################################

from package.scores.order import SCORES

scores_cuh = "#pragma once\n"
scores_cu = '#include "package/scores/scores.cuh"\n\n'

for scores in SCORES:
	scores_cuh += f'#include "package/scores/{scores}/head/{scores}.cuh"\n'

for scores in SCORES:
	exec(f"from package.scores.{scores}.py.{scores} import {scores.upper()}")

arrays = {
	"void* (*OPTI_SCORE_SPACE_MK_ARRAY[SCORES])(Opti_t * opti) = {" : "_space_mk",
	"void (*OPTI_SCORE_SET_ONE_ARG_ARRAY[SCORES])(Opti_t * opti, char * name, char * value) = {" : "_set_one_arg",
	"void (*OPTI_COMPUTE_LOSS_ARRAY[SCORES])(Opti_t * opti) = {" : "_loss",
	"void (*OPTI_SCORES_DLOSS_ARRAY[SCORES])(Opti_t * opti) = {" : "_dloss",
	"void (*OPTI_FREE_SCORE_ARRAY[SCORES])(Opti_t * opti) = {" : "_free",
	"const uint SCORE_CONST_AMOUNT[SCORES] = {" : "_CONSTS_AMOUNT",
	"const char ** SCORE_CONST_ARRAY[SCORES] = {" : "_CONST_ARRAY"
}

for key,value in arrays.items():
	scores_cu += key + '\n'
	for score in SCORES:
		scores_cu += '\t' + score.upper() + value + ',\n'
	scores_cu += "};\n\n"

with open("package/scores/scores.cu", "w") as co:
	co.write(scores_cu)

with open("package/scores/scores.cuh", "w") as co:
	co.write(scores_cuh)

########################################################################################################

########################################################################################################
############################################ META ######################################################
########################################################################################################

meta_cuh = f"""#pragma once

#define INSTS {len(INSTS)}

#define SCORES {len(SCORES)}
#define OPTIS {len(OPTIS)}
"""

with open("package/meta.cuh", "w") as co:
	co.write(meta_cuh)

########################################################################################################
############################################ PACKAGE.CUH ######################################################
########################################################################################################

package_cuh = f"""#pragma once

//	This includes Global #define for both kernel and package headers
#include "package/meta.cuh"

//	This includes all the kernel headers
//	It includes optis.cuh that includes train.cuh ...
#include "kernel/head/testpackage.cuh"

//	This include all the package headers
#include "package/insts/insts.cuh"
#include "package/optis/optis.cuh"
#include "package/scores/scores.cuh"

//	Arrays are declared in headers and writed in package/src/*.cu"""

with open("package/package.cuh", "w") as co:
	co.write(package_cuh)

########################################################################################################
############################################ PACKAGE.PY ######################################################
########################################################################################################

package_py = f'''######################### INSTS #########################

from package.insts.order import INSTS as INSTS_NAMES

INSTS_NAMES = [
	name.upper() for name in INSTS_NAMES
]

INSTS = []
INSTS_DICT = {{}}

for _id, inst_name in enumerate(INSTS_NAMES):
	exec(f"from package.insts.{{inst_name.lower()}}.py.{{inst_name.lower()}} import {{inst_name}}")
	exec(f"INSTS += [{{inst_name}}]")
	exec(f"INSTS_DICT['{{inst_name}}'] = {{inst_name}}")
	exec(f"{{inst_name}}._id = {{_id}}")
	exec(f"{{inst_name}}.ID = {{_id}}")

########################  SCORES ########################

from package.scores.order import SCORES

SCORES_NAMES = [
	name.upper() for name in SCORES
]

SCORES = []

for score in SCORES_NAMES:
	exec(f"from package.scores.{{score.lower()}}.py.{{score.lower()}} import {{score}}")
	exec(f"SCORES += [{{score}}]")

######################### OPTIS #########################

from package.optis.order import OPTIS

OPTIS_NAMES = [
	name.upper() for name in OPTIS
]

OPTIS = []

for opti in OPTIS_NAMES:
	exec(f"from package.optis.{{opti.lower()}}.py.{{opti.lower()}} import {{opti}}")
	exec(f"OPTIS += [{{opti}}]")

###################################################################################
###################################################################################
###################################################################################

####################### TEST_MODELS #####################

TEST_MODELS_NAMES = [
	("TEST_MDL_" + name) for name in INSTS_NAMES
]

TEST_MODELS = []

for inst_name in INSTS_NAMES:
	exec(f"from package.insts.{{inst_name.lower()}}.py.test_mdl_{{inst_name.lower()}} import TEST_MDL_{{inst_name.upper()}}")
	exec(f"TEST_MODELS += [TEST_MDL_{{inst_name.upper()}}]")

####################### TEST_SCORES #####################

TEST_SCORES_NAMES = [
	("TEST_SCORE_" + name) for name in SCORES_NAMES
]

TEST_SCORES = []

for score in SCORES_NAMES:
	exec(f"from package.scores.{{score.lower()}}.py.test_score_{{score.lower()}} import TEST_SCORE_{{score.upper()}}")
	exec(f"TEST_SCORES += [TEST_SCORE_{{score.upper()}}]")

####################### TEST_OPTIS #####################

TEST_OPTIS_NAMES = [
	("TEST_OPTI_" + name) for name in OPTIS_NAMES
]

TEST_OPTIS = []

for opti in OPTIS_NAMES:
	exec(f"from package.optis.{{opti.lower()}}.py.test_opti_{{opti.lower()}} import TEST_OPTI_{{opti.upper()}}")
	exec(f"TEST_OPTIS += [TEST_OPTI_{{opti.upper()}}]")'''

with open("package/package.py", "w") as co:
	co.write(package_py)