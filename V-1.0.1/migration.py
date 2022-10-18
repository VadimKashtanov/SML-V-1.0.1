'''
	C'est le scripte que j'ai utiliser pour transferer les fichiers de 0.32e a 1.0 qui est dans une autre architecture
'''

_from = "/home/vadim/Bureau/Simple ML V-0.x.x/V-0.32e/package"
to = "/home/vadim/Bureau/Simple ML V-0.x.x/V-1.0/package_prebuild"

from os import system, listdir

insts = listdir(_from + "/head/insts")
del insts[insts.index("Classification")]

#
#
#               Feed Forward Ony Genetic
#
#       Faire un systeme genetique que pour Feed Forward
# On va voire si le systeme qui permet d'imbriquer les insts doit etre fait directement
# Pour un systeme avec autant de input qu'on veut et pas limité a feed-forward

for inst in insts:
	out_path = f"{to}/insts/{inst}"
		
	system(f'rm -r "{out_path}"')
	system(f'mkdir "{out_path}"')
	system(f'mkdir "{out_path}/head" "{out_path}/src" "{out_path}/py"')
		
	system(f'cp "{_from}/head/insts/{inst}/"*.cuh "{out_path}/head"')
	system(f'cp "{_from}/py/insts/{inst}.py" "{out_path}/py"')
	system(f'cp "{_from}/py/testpackage/test_mdls/test_mdl_{inst}.py" "{out_path}/py"')
	system(f'cp -r "{_from}/src/insts/{inst}/"* "{out_path}/src"')

optis = listdir(_from + "/head/optis/optis")
optis = [i.replace('.cuh', '') for i in optis]

for opti in optis:
	out_path = f"{to}/optis/{opti}"
		
	system(f'rm -r "{out_path}"')
	system(f'mkdir "{out_path}"')
	system(f'mkdir "{out_path}/head" "{out_path}/src" "{out_path}/py"')
	
	system(f'cp "{_from}/head/optis/optis/{opti}.cuh" "{out_path}/head"')
	system(f'cp "{_from}/py/optis/{opti}.py" "{out_path}/py"')
	system(f'cp "{_from}/py/testpackage/test_optis/test_{opti}.py" "{out_path}/py"')
	system(f'cp -r "{_from}/src/optis/optis/{opti}/"* "{out_path}/src"')
		
scores = listdir(_from + "/head/optis/scores")
scores = [i.replace('.cuh', '') for i in scores]
del scores[scores.index("scores_etc")]

for score in scores:
	out_path = f"{to}/scores/{score}"
		
	system(f'rm -r "{out_path}"')
	system(f'mkdir "{out_path}"')
	system(f'mkdir "{out_path}/head" "{out_path}/src" "{out_path}/py"')
		
	system(f'cp "{_from}/head/optis/scores/{score}.cuh" "{out_path}/head"')
	system(f'cp "{_from}/py/scores/{score}.py" "{out_path}/py"')
	system(f'cp "{_from}/py/testpackage/test_scores/test_{score}.py" "{out_path}/py"')
	system(f'cp -r "{_from}/src/optis/scores/{score}/"* "{out_path}/src"')