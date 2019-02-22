from Models.HH_Test.model import model
import pickle as pkl

modelDict = {
	'model':model,
	'name':'HH_Test',
	'equil_inits':[
		0.,
		0.,
		0.,
		0.
	],
	'inits':[
		0.,
		0.,
		0.,
		0.
	],
	'colDict':{
		'V':0,
		'n':1,
		'm':2,
		'h':3
	}
}

with open("./Models/HH_Test/modelDict.pkl", "wb") as f:
	pkl.dump(modelDict, f)