from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from django.template import loader

from collections import OrderedDict

import model

def index(request):
	return render_to_response("TR/index.html")

def predict(request):
	template = loader.get_template('TR/predict.html')
	features_names = ['seed_chi2PerDoF','seed_p','seed_pt','seed_nLHCbIDs','seed_nbIT','seed_nLayers','seed_x','seed_y','seed_tx','seed_ty']
	features = OrderedDict()
	for key in features_names:
		features[key] = float(request.GET[key])
	result = 'True :D' if model.model.pred(*features.values()) else 'False :('
	context = {'features': features, 'sum': result}
	return HttpResponse(template.render(context, request))
