#include "functions.h"


Neuron::Neuron(int nin) {

	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_real_distribution<float> distr(-1.0f, 1.0f);


	for (int i = 0; i < nin; i++) {
		w.push_back(new Value(std::make_pair<Value*, Value*>(nullptr, nullptr), distr(eng)));
	}

	b = new Value(std::make_pair<Value*, Value*>(nullptr, nullptr), distr(eng));
}

Value* Neuron::forward(std::vector<Value*>& x) {
	Value* r = new Value(std::make_pair<Value*, Value*>(nullptr, nullptr), 0.0f);

	for (int i = 0; i < x.size(); i++) {

		r = add(r, mul(w[i], x[i]));//Using full value passing, test this
	}

	return tanh(r);
}

Layer::Layer(int nin, int nout) {
		
	for (int i = 0; i < nout; i++) {
		neurons.push_back(Neuron(nin));
	}
}


std::vector<Value*> Layer::forward(std::vector<Value*> &x) {

	std::vector<Value*> outs(neurons.size(), nullptr);

	for (int i = 0; i < neurons.size(); i++) {
		outs[i] = neurons[i].forward(x);
	}

	return outs;
}


MLP::MLP(int nin, std::vector<float> nouts) {
	std::vector<float> sz = nouts;
	sz.insert(sz.begin(), nin);

	for (int i = 0; i < nouts.size(); i++) {
		layers.push_back(Layer(sz[i], sz[i + 1]));
	}
}


std::vector<Value*> MLP::forward(std::vector<float>& x) {

	std::vector<Value*> r(x.size(), nullptr);
	for (int i = 0; i < x.size(); i++) {
		r[i] = new Value(std::make_pair<Value*, Value*>(nullptr, nullptr), x[i]);
	}

	for (auto &x : layers) {
		r = x.forward(r);
	}
	return r;
}


std::vector<Value*> Neuron::paramaters() {
	std::vector<Value*> t = w;
	t.insert(t.begin(), b);

	return t;
}


std::vector<Value*> Layer::parameters() {
	std::vector<Value*> params;
	for (auto& n : neurons) {

		std::vector<Value*> pm = n.paramaters();
		params.insert(params.end(), pm.begin(), pm.end());
	}

	return params;
}	
std::vector<Value*> MLP::parameters() {
	std::vector <Value*> r;

	for (auto& l : layers) {
		std::vector<Value*> t = l.parameters();
		r.insert(r.begin(), t.begin(), t.end());
	}

	return r;
}

MLP::~MLP(){
	//This deconstructor is only for my implementation of C++
	//Since I use heap memory, I have to free the memory
	//It would be clunky to use unique pointers due to the disgusting syntax similar to std::pair the devil
	//std::make_pair<Fuck, Off> (Please);

	std::cout << "Deconstructed Network" << std::endl;
	for (auto & x : parameters()) {
		del(x->prev.first);
		del(x->prev.second);
		del(x->out);
		delete(x);
		//thorough :):)
	}

	//For some reason i feel as if this is doing nothing 
}
