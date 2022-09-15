#include "functions.h"

Value::Value(std::pair<Value*, Value*> children, const float &d) : out(nullptr) {
	data = d;
	prev = children;
}

void Value::_backward() {

	if (out != nullptr) {	
		Value* other = this == out->prev.first ? out->prev.second : out->prev.first;

		switch (out->type) {
		case '+':
			grad += out->grad;
			break;
		case '*':

			grad += other->data * out->grad;
			break;
		case 's':
			//sigmoid
			grad += d_sigmoid(data) * out->grad;
			break;
		case 't':
			//tanh
			grad += d_tanh(data) * out->grad;
			break;
		case 'p':
			//I honestly do not care about powers which are not 2 so if you arre reading this and want it you can go do it:nerdface:
			grad += 2.0f * data * out->grad;
			break;
		}

	}	
}

void Value::backward() {
	
	std::vector<Value*> topo;
	std::set<Value*> s;
	build_topo(this, s, topo);

	grad = 1;

	for (int i = topo.size() - 2; i >= 0; i--) {
		topo[i]->_backward();
	}
}
