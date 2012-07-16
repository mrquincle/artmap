/**
 * @brief Test ARTMAP using supervised learning task
 * @file art_test.cpp
 *
 * This file is created at Almende B.V. It is open-source software and part of the Common
 * Hybrid Agent Platform (CHAP). A toolbox with a lot of open-source tools, ranging from
 * thread pools and TCP/IP components to control architectures and learning algorithms.
 * This software is published under the GNU Lesser General Public license (LGPL).
 *
 * It is not possible to add usage restrictions to an open-source license. Nevertheless,
 * we personally strongly object against this software used by the military, in the
 * bio-industry, for animal experimentation, or anything that violates the Universal
 * Declaration of Human Rights.
 *
 * Copyright © 2012 Anne van Rossum <anne@almende.com>
 *
 * @author     Anne C. van Rossum
 * @date       Apr 23, 2012
 * @project    Replicator FP7
 * @company    Almende B.V.
 * @case       Machine learning (fit for Surveyeor robots)
 */

#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <assert.h>
#include <artMap.h>
#include <art.h>

#if (RUNONPC==true)
#include <DataDecorator.h>
#include <Plot.h>
#endif

using namespace std;
using namespace almendeSensorFusion;

enum TestCase { TC_CIRCLE, TC_HALVES, TC_COUNT };

TestCase testCase = TC_CIRCLE;

/**
 * Input: expects x in [0,1], y in [0,1]
 * Will return if
 */
bool withinCircle(float x, float y) {
	return ((2*x-1)*(2*x-1)+(2*y-1)*(2*y-1) < 1.0);
}

bool verticalHalves(float x, float y) {
	return (x < 0.5);
}


void getRandomSample(ART_ASPECT *aspect, ART_TYPE &class_id) {
	float x = (float)drand48();
	float y = (float)drand48();
	float cl;
	switch (testCase) {
	case TC_CIRCLE:
		cl = withinCircle(x,y) ? 1 : 0;
		break;
	case TC_HALVES:
		cl = verticalHalves(x,y) ? 1 : 0;
		break;
	default:
		cl = verticalHalves(x,y) ? 1 : 0;
		break;
	}
	aspect->clear();
	aspect->push_back(x);
	aspect->push_back(y);
	aspect->push_back(1.0-x);
	aspect->push_back(1.0-y);
	class_id = cl;
}

int main(int argc, char *argv[]) {
	cout << "Test for ARTMAP" << endl;
	srand48( time(NULL) );
	Art &input = *new Art(false, true, true);
	Art &supervisor = *new Art(false, true, true);
	input.setVigilance(0.80);
	input.setNetworkReliability(0.8);
	supervisor.setVigilance(0.99);
	supervisor.setNetworkReliability(1.0);

	std::vector<Art*> &networks = *new std::vector<Art*>();
	networks.push_back(&input);
	networks.push_back(&supervisor);

	ArtMap *artmap = new ArtMap(&networks);

	ART_ASPECT aspect;
	ART_TYPE class_id;
	ART_DISTRIBUTED_CLASS cl;

	ART_VIEW inputVector;
	//	int found = 0;
	//	vector<int> winnCount(0);

	// we only have one "view" for ARTMAP (at t=now).
	ART_VIEWS multipleInputVector(0);

	ART_DISTRIBUTED_CLASSES *output;

	int mis_classified = 0, correct_classified = 0;
	int N = 100000;

#if (RUNONPC==true)
	cout << "We will plot the results as .ppm file" << endl;
	int L = 256;
	float *data = new float[L*L];
	for (int i = 0; i < L; ++i) {
		for (int j = 0; j < L; ++j) {
			data[i+j*L] = 0;
		}
	}
#else
	cout << "We will not plot anything" << endl;
#endif
	for (int t = 0; t < N; ++t) {
		getRandomSample(&aspect, class_id);
		cl.clear();
		cl.push_back(class_id); // no complement-encoding here!
		inputVector.clear();
		inputVector.push_back(&aspect);

		if (t < N*0.9) {
			inputVector.push_back(&cl);
			output = artmap->classify(inputVector);
			for (int x = 0; x < output->size(); ++x) {
				if((*output)[x] != NULL)
					delete (*output)[x];
			}
			delete output;

		} else {
			inputVector.push_back(NULL);
			output = artmap->classify(inputVector);
			if(output->size() > 1 && ((*output)[1]) != NULL) {
				ART_TYPE foundClass = (*((*output)[1]))[0];
				PROTOTYPE* prot = supervisor.getPrototype(foundClass);
				ART_TYPE cl_id = (*prot)[0];
//				cout << "Found class id: " << cl_id << " (with input having # prototypes : ";
//				cout << input.getF2()->size() << ")" << endl;
				if(cl_id == class_id)
					correct_classified++;
				else
					mis_classified++;
#if (RUNONPC==true)
				int x = L * aspect[0];
				int y = L * aspect[1];
				float p = (0.99-(cl_id*0.5));
				data[x+y*L]=p;
				data[(x+1)%L+y*L]=p;
				data[x+((y+1)%L)*L]=p;
				data[(x+1)%L+((y+1)%L)*L]=p;
#endif
			}
		}
	}
	cout << "Number of prototypes necessary: " << input.getF2()->size() << "" << endl;
	cout << "Classified [correct/incorrect]: [" << correct_classified << "/" << mis_classified << "]" << endl;
	delete artmap;

#if (RUNONPC==true)
	Plot *p = new Plot();
	string f = "artmap";
	switch (testCase) {
	case TC_CIRCLE:
		f += "_circle";
		break;
	case TC_HALVES:
		f += "_halves";
		break;
	default:
		break;
	}
	p->SetFileName(f, PPM);
	cout << "Write to file: " << f << endl;
	DataContainer &dc = p->GetData();
	dc.SetData(data, L*L);
	p->Draw(PPM);
	delete [] data;
#endif

	return EXIT_SUCCESS;
}






